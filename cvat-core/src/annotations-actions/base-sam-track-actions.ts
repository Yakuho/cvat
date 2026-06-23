// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { throttle } from 'lodash';
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
    batch: Array<{
        frame: number;
        objectId: string;
        labelId: number;
        type: string;
        non_cond: Record<string, { id: number }>;
        cond: Record<string, { id: number }>;
    }>;
}

export type SAMTrackActionOutput = Array<{
    frame: number;
    confidence: number;
    created: SerializedShape | SerializedTrack | null;
}>;

export abstract class BaseSAMTrackAction extends BaseAction {
    public frameFrom: number;
    public frameTo: number;
    public isRenderActionRunnerFrames(): boolean { return false; }
    public abstract run(input: SAMTrackActionInput): Promise<SAMTrackActionOutput>;
    public abstract applyFilter(
        input: Pick<SerializedCollection, 'shapes' | 'tracks'>
    ): Pick<SerializedCollection, 'shapes' | 'tracks'>;
}

function ProcessShape(
    shape: SerializedShape,
    removeFrameIds: Array<number>,
    INTracksRecord: Record<number, Record<number, SerializedShape | SerializedTrack>>,
    INDependRecord: Array<Record<number, SerializedShape | SerializedTrack>>,
): void {
    if ('track_id' in shape) {
        const trackID = shape.track_id as number; // TODO: track_id attribute maybe support future
        const trackFrames = INTracksRecord[trackID] ?? {};

        if (trackFrames[shape.frame]) {
            throw new Error(
                `Invalid annotation: multiple objects with track_id=${trackID} found in frame=${shape.frame}. ` +
                'A track_id must be unique within a single frame.',
            );
        }

        if (!removeFrameIds.includes(shape.frame)) {
            trackFrames[shape.frame] = shape;
            INTracksRecord[trackID] = trackFrames;
        }
    } else if (!removeFrameIds.includes(shape.frame)) {
        INDependRecord.push({ [shape.frame]: shape });
    }
}

function ProcessTrack(
    track: SerializedTrack,
    action: BaseSAMTrackAction,
    removeFrameIds: Array<number>,
    INTracksRecord: Record<number, Record<number, SerializedShape | SerializedTrack>>,
    INDependRecord: Array<Record<number, SerializedShape | SerializedTrack>>,
): void {
    if ('track_id' in track) {
        const trackID = track.track_id as number; // TODO: track_id attribute maybe support future
        const trackFrames = INTracksRecord[trackID] ?? {};

        for (let { frame } = track.shapes[0]; frame <= action.frameTo; frame++) {
            if ((track.shapes.filter((kf) => kf.frame <= frame).at(-1)).outside) break;
            if (trackFrames[frame]) {
                throw new Error(
                    `Invalid annotation: multiple objects with track_id=${trackID} found in frame=${frame}. ` +
                    'A track_id must be unique within a single frame.',
                );
            }
            if (!removeFrameIds.includes(frame)) {
                trackFrames[frame] = track;
            }
        }

        if (Object.keys(trackFrames).length > 0) {
            INTracksRecord[trackID] = trackFrames;
        }
    } else {
        const trackRecord: Record<number, SerializedTrack> = {};

        for (let { frame } = track.shapes[0]; frame <= action.frameTo; frame++) {
            if ((track.shapes.filter((kf) => kf.frame <= frame).at(-1)).outside) break;
            if (!removeFrameIds.includes(frame)) {
                trackRecord[frame] = track;
            }
        }

        if (Object.keys(trackRecord).length > 0) {
            INDependRecord.push(trackRecord);
        }
    }
}

class SAMTrackObject {
    private order: Array<number>;
    private readonly removeFrameIds: Array<number>;

    private readonly type: string;
    private readonly labelId: number;
    private readonly objectId: string;
    private readonly non_cond: Record<number, SerializedShape | SerializedTrack>;
    private readonly cond: Record<number, SerializedShape | SerializedTrack>;

    private readonly frameFrom: number;
    private readonly frameTo: number;

    private updateOrder(): void {
        const order: number[] = [];
        const frames: number[] = [
            ...Object.keys(this.cond).map(Number),
            ...Object.keys(this.non_cond).map(Number),
            ...this.removeFrameIds,
        ].sort((a, b) => a - b);

        const processGroup = (group: number[]): void => {
            if (group.every((id) => this.removeFrameIds.includes(id))) return;
            for (const frame of [Math.min(...group) - 1, Math.max(...group) + 1]) {
                if (frame >= this.frameFrom && frame <= this.frameTo && !order.includes(frame)) {
                    order.push(frame);
                }
            }
        };

        if (frames.length === 0) return;

        let current: number[] = [frames[0]];
        for (let i = 1; i < frames.length; i++) {
            if (frames[i] === frames[i - 1] + 1) {
                current.push(frames[i]);
            } else {
                processGroup(current);
                current = [frames[i]];
            }
        }
        processGroup(current);
        this.order = order;
    }

    private getObjectId(data: Record<number, SerializedShape | SerializedTrack>): string {
        const pointsCluster = Object.values(data).flatMap((item) => (
            'shapes' in item ? item.shapes.flatMap((shape) => shape.points) : item.points
        )).map(String).join(',');

        /*
            Using simple hash algorithm: djb2

            Why 5381?
                Daniel J. Bernstein designed djb2 and, after extensive testing of initial values,
                found that 5381 combined with the formula hash * 33 ^ c yielded the fewest collisions on real-world string data.
         */
        let hash = 5381;
        for (let i = 0; i < pointsCluster.length; i++) {
            hash = ((hash << 5) + hash) ^ pointsCluster.charCodeAt(i);
            hash |= 0;
        }
        return (hash >>> 0).toString(16);
    }

    private getLabelId(data: Record<number, SerializedShape | SerializedTrack>): number {
        const labels = Object.values(data).map((item) => item.label_id);

        if (!labels.length) {
            throw new Error('No objects selected');
        }

        if (!labels.every((l) => l === labels[0])) {
            throw new Error('Selected objects have different label');
        }

        return labels[0];
    }

    private getType(data: Record<number, SerializedShape | SerializedTrack>): string {
        const types = Object.values(data).map((item) => ('shapes' in item ? item.shapes[0].type : item.type));

        if (!types.length) {
            throw new Error('No objects selected');
        }

        if (!types.every((t) => t === types[0])) {
            throw new Error('Selected objects have different type');
        }

        return types[0];
    }

    get length(): number {
        return (this.frameTo - this.frameFrom) - this.removeFrameIds.length -
            Object.keys({ ...this.cond, ...this.non_cond }).length;
    }

    generate(): {
        frame: number;
        objectId: string;
        labelId: number;
        type: string;
        non_cond: Record<string, { id: number }>;
        cond: Record<string, { id: number }>;
    } {
        if (this.length === 0) return;
        if (this.order.length === 0) {
            this.updateOrder();
            if (this.order.length === 0) return;
        }

        return {
            frame: this.order.pop() as number,
            objectId: this.objectId,
            labelId: this.labelId,
            type: this.type,
            non_cond: Object.fromEntries(
                Object.entries(this.non_cond).map(([k, v]) => [k, {
                    id: v ? v.id : null,
                    type: v ? ('shapes' in v ? 'track' : 'shape') : null,
                }]),
            ),
            cond: Object.fromEntries(
                Object.entries(this.cond).map(([k, v]) => [k, {
                    id: v.id as number,
                    type: 'shapes' in v ? 'track' : 'shape',
                }]),
            ),
        };
    }

    update(
        frame: number,
        created: SerializedShape | SerializedTrack | null,
    ): void { this.non_cond[frame] = created; }

    constructor(
        data: Record<number, SerializedShape | SerializedTrack>,
        removeFrameIds: Array<number>,
        frameFrom: number,
        frameTo: number,
    ) {
        this.removeFrameIds = removeFrameIds;
        this.frameFrom = frameFrom;
        this.frameTo = frameTo;
        this.cond = data;

        this.type = this.getType(data);
        this.labelId = this.getLabelId(data);
        this.objectId = this.getObjectId(data);
        this.non_cond = {};
        this.order = [];
    }
}

async function execute(
    instance: Job | Task,
    action: BaseSAMTrackAction,
    actionParameters: Record<string, string>,
    callback: (removeFrameIds: Array<number>) => Promise<Array<SAMTrackObject>>,
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
    const matchBatch = (samTrackObjects: Array<SAMTrackObject>, batchSize: number): number[][] => {
        const result: number[][] = [];
        const tracksNumbsRecord = samTrackObjects.map((v) => v.length);

        // eslint-disable-next-line no-param-reassign
        if (batchSize <= 0) batchSize = Math.max(...tracksNumbsRecord);

        while (tracksNumbsRecord.some((v) => v > 0)) {
            const batch: number[] = [];
            for (let i = 0; i < tracksNumbsRecord.length; i++) {
                if (tracksNumbsRecord[i] > 0) {
                    batch.push(i);
                    tracksNumbsRecord[i] -= 1;
                    if (batch.length === batchSize) break;
                }
            }
            result.push(batch);
        }
        return result;
    };

    try {
        await showMessageWithPause('Action initialization', 0, 500);
        if (cancelled()) {
            return;
        }

        // Initial Action Setting
        await action.init(instance, prepareActionParameters(action.parameters, actionParameters));
        if (typeof action.frameFrom !== 'number' || typeof action.frameTo !== 'number' || action.frameFrom >= action.frameTo) {
            await showMessageWithPause('No frames or invalid frames to process', 100, 1500);
            return;
        }

        const batchSize = -1; // TODO: Maybe read from SAMFunction spec. -1 for all
        const removeFrameIds: Array<number> = (await Promise.all(
            Array.from({ length: action.frameTo - action.frameFrom }, (_, i) => action.frameFrom + i)
                .map(async (frame) => {
                    const frameData = await Object.getPrototypeOf(instance).frames
                        .get.implementation.call(instance, frame);
                    return frameData.deleted ? frame : null;
                }),
        )).filter((frame: number): boolean => frame !== null);

        // Callback call must be late for action.init, because callback will get action attribute: frameFrom, frameTo
        const samTrackObjects: Array<SAMTrackObject> = await callback(removeFrameIds);
        if (samTrackObjects.length === 0) {
            await showMessageWithPause('No Annotations to process', 100, 1500);
            return;
        }

        // Iterate SAM Video Tracking
        await showMessageWithPause('Actions are running', 0, 500);
        const samTrackPredBatch = matchBatch(samTrackObjects, batchSize);
        const sanTrackPredIters = samTrackPredBatch.length;
        for (let i = 0; i < sanTrackPredIters; i++) {
            const samTrackObjectsBatch = samTrackPredBatch[i].map((gid) => samTrackObjects[gid]);

            // NOTE: Maybe confidence need to save in future
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            for (const [bid, { frame, created, confidence }] of (await action.run({
                batch: samTrackObjectsBatch.map((obj) => obj.generate()),
                onProgress: decoratedOnProgress,
                cancelled,
            })).entries()) {
                samTrackObjectsBatch[bid].update(frame, created); // update state

                if (created !== null) {
                    await instance.annotations.commit(
                        { shapes: [created as SerializedShape], tags: [], tracks: [] },
                        { shapes: [], tags: [], tracks: [] },
                        frame,
                    );
                }
            }

            const progress = Math.ceil(((i + 1) / sanTrackPredIters) * 100);
            decoratedOnProgress('Actions are running', progress);
            if (cancelled()) {
                return;
            }
        }

        await showMessageWithPause('Committing handled objects', 100, 1500);
        if (cancelled()) {
            return;
        }
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
        async (removeFrameIds: Array<number>): Promise<Array<SAMTrackObject>> => {
            const INTracksRecord: Record<number, Record<number, SerializedShape | SerializedTrack>> = {};
            const INDependRecord: Array<Record<number, SerializedShape | SerializedTrack>> = [];

            const exportedCollection = getCollection(instance).export();
            validateClientIDs(exportedCollection);

            const filteredByAction = action.applyFilter({
                shapes: exportedCollection.shapes,
                tracks: exportedCollection.tracks,
            });

            const annotationsFilter = new AnnotationsFilter();
            const filteredClientIDs = annotationsFilter.filterSerializedCollection({
                tags: [],
                shapes: filteredByAction.shapes,
                tracks: filteredByAction.tracks,
            }, instance.labels, filters);

            for (const shape of filteredByAction.shapes) {
                if (!filteredClientIDs.shapes.includes(shape.clientID) || !shape.id) continue;
                ProcessShape(shape, removeFrameIds, INTracksRecord, INDependRecord);
            }
            for (const track of filteredByAction.tracks) {
                if (!filteredClientIDs.tracks.includes(track.clientID) || !track.id) continue;
                ProcessTrack(track, action, removeFrameIds, INTracksRecord, INDependRecord);
            }

            return [...INDependRecord, ...Object.values(INTracksRecord)].map(
                (record) => new SAMTrackObject(record, removeFrameIds, action.frameFrom, action.frameTo)).filter(
                (obj: SAMTrackObject): boolean => obj.length > 0,
            );
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
        async (removeFrameIds: Array<number>): Promise<Array<SAMTrackObject>> => {
            const INTracksRecord: Record<number, Record<number, SerializedShape | SerializedTrack>> = {};
            const INDependRecord: Array<Record<number, SerializedShape | SerializedTrack>> = [];
            const exported = await Promise.all(states.map((s) => s.export()));

            const shapes: SerializedShape[] = [];
            const tracks: SerializedTrack[] = [];
            exported.forEach((state, idx) => {
                if (!state.id) throw new Error('The currently selected annotation has not been saved.');

                if (states[idx].objectType === ObjectType.SHAPE) {
                    shapes.push(state as SerializedShape);
                } else if (states[idx].objectType === ObjectType.TRACK) {
                    tracks.push(state as SerializedTrack);
                }
            });
            const filteredByAction = action.applyFilter({ shapes, tracks });

            for (const shape of filteredByAction.shapes) {
                ProcessShape(shape, removeFrameIds, INTracksRecord, INDependRecord);
            }

            for (const track of filteredByAction.tracks) {
                ProcessTrack(track, action, removeFrameIds, INTracksRecord, INDependRecord);
            }

            return [...INDependRecord, ...Object.values(INTracksRecord)].map(
                (record) => new SAMTrackObject(record, removeFrameIds, action.frameFrom, action.frameTo)).filter(
                (obj: SAMTrackObject): boolean => obj.length > 0,
            );
        },
        onProgress,
        cancelled,
    );
}
