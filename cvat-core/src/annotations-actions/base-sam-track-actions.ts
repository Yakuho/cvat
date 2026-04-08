// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { throttle, range } from 'lodash';
import AnnotationsFilter from '../annotations-filter';
import { Job, Task } from '../session';
import { SerializedCollection, SerializedShape, SerializedTrack } from '../server-response-types';
import { EventScope, ObjectType } from '../enums';
import { getCollection } from '../annotations';
import { BaseAction, prepareActionParameters, validateClientIDs } from './base-action';
import ObjectState from '../object-state';

export interface SAMTrackActionInput {
    onProgress(message: string, percent: number): void;
    cancelled(): boolean;
    collection: Pick<SerializedCollection, 'shapes' | 'tracks'>;
    frameData: {
        width: number;
        height: number;
        number: number;
    };
}

export interface SAMTrackActionOutput {
    created: SAMTrackActionInput['collection'];
    deleted: SAMTrackActionInput['collection'];
}

export abstract class BaseSAMTrackAction extends BaseAction {
    public frameFrom: number;
    public frameTo: number;
    public isRenderActionRunnerFrames(): boolean { return false; }
    public abstract run(input: SAMTrackActionInput): Promise<SAMTrackActionOutput>;
    public abstract applyFilter(
        input: Pick<SAMTrackActionInput, 'collection' | 'frameData'>
    ): SAMTrackActionInput['collection'];
}

async function execute(
    instance: Job | Task,
    action: BaseSAMTrackAction,
    actionParameters: Record<string, string>,
    callback: () => Promise<{
        filteredShapesByFrame: Record<number, SerializedShape[]>;
        filteredTracksByFrame: Record<number, SerializedTrack[]>;
    }>,
    onProgress: (message: string, progress: number) => void,
    cancelled: () => boolean,
): Promise<void> {
    const event = await instance.logger.log(EventScope.annotationsAction, { name: action.name }, true);

    const decoratedOnProgress = throttle(onProgress, 100, { leading: true, trailing: true });
    const showMessageWithPause = async (message: string, progress: number, duration: number): Promise<void> => {
        // wrapper that gives a chance to abort action
        decoratedOnProgress(message, progress);
        await new Promise((resolve) => { setTimeout(resolve, duration); });
    };
    const validateUniqueLabels = (
        shapes: Record<number, SerializedShape[]>,
        tracks: Record<number, SerializedTrack[]>): void => {
        const allCondFrames = Array.from(
            new Set([...Object.keys(shapes), ...Object.keys(tracks)]),
        ).map(Number).sort((a, b) => a - b);
        for (const frame of allCondFrames) {
            const frameShapes = shapes[frame] ?? [];
            const frameTracks = tracks[frame] ?? [];
            const frameLabelsIDs = [
                ...frameShapes.map((s) => s.label_id),
                ...frameTracks.map((t) => t.label_id),
            ];

            const labelMap = new Map(instance.labels.map((label) => [label.id, label.name]));
            const countMap = frameLabelsIDs.reduce((map, id) => {
                map.set(id, (map.get(id) ?? 0) + 1);
                return map;
            }, new Map<number, number>());

            const invalidLabels = Array.from(countMap.entries())
                .filter(([, count]) => count > 1)
                .map(([id]) => labelMap.get(id));

            if (invalidLabels.length > 0) {
                throw new Error(
                    `Duplicate labels detected on frame ${frame}: [${invalidLabels.join(', ')}]. ` +
                    'Each label must appear at most once per frame, ' +
                    'otherwise the tracker cannot distinguish between objects of the same category.',
                );
            }
        }
    };

    try {
        await showMessageWithPause('Action initialization', 0, 500);
        if (cancelled()) {
            return;
        }

        await action.init(instance, prepareActionParameters(action.parameters, actionParameters));
        if (typeof action.frameFrom !== 'number' || typeof action.frameTo !== 'number' || action.frameFrom >= action.frameTo) {
            await showMessageWithPause('No frames or invalid frames to process', 100, 1500);
            return;
        }

        // callback must be late for action.init, because callback will take action attribute: frameFrom, frameTo
        const { filteredShapesByFrame, filteredTracksByFrame } = await callback();

        // Check unique label per frame to prevent predict problem
        validateUniqueLabels(filteredShapesByFrame, filteredTracksByFrame);

        const totalUpdates = { created: { shapes: [], tracks: [] }, deleted: { shapes: [], tracks: [] } };
        // Iterate over frames
        const allFrameNumbers = instance instanceof Job ?
            await instance.frames.frameNumbers() : range(0, instance.size);
        const frameNumbers = allFrameNumbers.filter((frame) => frame >= action.frameFrom && frame <= action.frameTo);
        const totalFrames = frameNumbers.length;

        const hasNoShapes = Object.keys(filteredShapesByFrame).length === 0;
        const hasNoTracks = Object.keys(filteredTracksByFrame).length === 0;
        if (totalFrames === 0 || (hasNoShapes && hasNoTracks)) {
            await showMessageWithPause('No frames/annotations to process', 100, 1500);
            return;
        }
        for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
            const frame = frameNumbers[frameIdx];
            const frameData = await Object.getPrototypeOf(instance).frames
                .get.implementation.call(instance, frame);

            // Ignore deleted frames
            if (!frameData.deleted) {
                const frameShapes = filteredShapesByFrame[frame] ?? [];
                const frameTracks = filteredTracksByFrame[frame] ?? [];

                // finally apply the own filter of the action
                const filteredByAction = action.applyFilter({
                    collection: {
                        shapes: frameShapes,
                        tracks: frameTracks,
                    },
                    frameData,
                });
                validateClientIDs(filteredByAction);

                const { created, deleted } = await action.run({
                    onProgress: decoratedOnProgress,
                    cancelled,
                    collection: {
                        shapes: filteredByAction.shapes,
                        tracks: filteredByAction.tracks,
                    },
                    frameData: {
                        width: frameData.width,
                        height: frameData.height,
                        number: frameData.number,
                    },
                });

                Array.prototype.push.apply(totalUpdates.created.shapes, created.shapes);
                Array.prototype.push.apply(totalUpdates.created.tracks, created.tracks);
                Array.prototype.push.apply(totalUpdates.deleted.shapes, deleted.shapes);
                Array.prototype.push.apply(totalUpdates.deleted.tracks, deleted.tracks);

                const progress = Math.ceil(((frameIdx + 1) / totalFrames) * 100);
                decoratedOnProgress('Actions are running', progress);
                if (cancelled()) {
                    return;
                }
            }
        }

        await showMessageWithPause('Committing handled objects', 100, 1500);
        if (cancelled()) {
            return;
        }

        await instance.annotations.commit(
            { shapes: totalUpdates.created.shapes, tags: [], tracks: totalUpdates.created.tracks },
            { shapes: totalUpdates.deleted.shapes, tags: [], tracks: totalUpdates.deleted.tracks },
            frameNumbers[0],
        );

        event.close();
    } finally {
        await action.destroy();
    }
}

export async function run(
    instance: Job | Task,
    action: BaseSAMTrackAction,
    actionParameters: Record<string, string>,
    filters: object[],
    onProgress: (message: string, progress: number) => void,
    cancelled: () => boolean,
): Promise<void> {
    await execute(
        instance,
        action,
        actionParameters,
        async () => {
            const exportedCollection = getCollection(instance).export();
            validateClientIDs(exportedCollection);

            const annotationsFilter = new AnnotationsFilter();
            const filteredClientIDs = annotationsFilter.filterSerializedCollection({
                shapes: exportedCollection.shapes,
                tags: [],
                tracks: exportedCollection.tracks,
            }, instance.labels, filters);
            const filteredShapesByFrame = exportedCollection.shapes.reduce((acc, shape) => {
                if (!filteredClientIDs.shapes.includes(shape.clientID) || !shape.id) return acc;

                if (shape.frame >= action.frameFrom && shape.frame <= action.frameTo) {
                    acc[shape.frame] = acc[shape.frame] ?? [];
                    acc[shape.frame].push(shape);
                }

                return acc;
            }, {} as Record<number, SerializedShape[]>);
            const filteredTracksByFrame = exportedCollection.tracks.reduce((acc, track) => {
                if (!filteredClientIDs.tracks.includes(track.clientID) || !track.id) return acc;

                for (let frame = action.frameFrom; frame <= action.frameTo; frame++) {
                    const prevKeyframe = track.shapes.filter((kf) => kf.frame <= frame).at(-1);
                    if (!prevKeyframe || prevKeyframe.outside) continue;
                    acc[frame] = acc[frame] ?? [];
                    acc[frame].push(track);
                }

                return acc;
            }, {} as Record<number, SerializedTrack[]>);
            return { filteredShapesByFrame, filteredTracksByFrame };
        },
        onProgress,
        cancelled,
    );
}

export async function call(
    instance: Job | Task,
    action: BaseSAMTrackAction,
    actionParameters: Record<string, string>,
    states: ObjectState[],
    onProgress: (message: string, progress: number) => void,
    cancelled: () => boolean,
): Promise<void> {
    await execute(
        instance,
        action,
        actionParameters,
        async () => {
            const filteredShapesByFrame: Record<number, SerializedShape[]> = {};
            const filteredTracksByFrame: Record<number, SerializedTrack[]> = {};

            const exported = await Promise.all(states.map((s) => s.export()));
            const exportedCollection = getCollection(instance).export();
            validateClientIDs(exportedCollection);

            // TODO: Should select all tracks and shapes which match current state label_id
            exported.forEach((state, idx) => {
                const { objectType } = states[idx];
                if (objectType === ObjectType.SHAPE) {
                    const shape = state as SerializedShape;
                    if (!shape.id) throw new Error('The currently selected shape object has not been saved.');
                    for (const s of exportedCollection.shapes) {
                        if (s.frame >= action.frameFrom && s.frame <= action.frameTo && s.label_id === shape.label_id) {
                            filteredShapesByFrame[s.frame] = filteredShapesByFrame[s.frame] ?? [];
                            filteredShapesByFrame[s.frame].push(s);
                        }
                    }
                } else if (objectType === ObjectType.TRACK) {
                    const track = state as SerializedTrack;
                    if (!track.id) throw new Error('The currently selected track object has not been saved.');
                    for (let frame = action.frameFrom; frame <= action.frameTo; frame++) {
                        const prevKeyframe = track.shapes.filter((kf) => kf.frame <= frame).at(-1);
                        if (!prevKeyframe || prevKeyframe.outside) continue;
                        filteredTracksByFrame[frame] = filteredTracksByFrame[frame] ?? [];
                        filteredTracksByFrame[frame].push(track);
                    }
                }
            });

            return { filteredShapesByFrame, filteredTracksByFrame };
        },
        onProgress,
        cancelled,
    );
}
