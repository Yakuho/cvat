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
        non_cond: Array<number>;
        cond: Record<number, { id: number, type: 'track' | 'shape' }>;
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

type SAMTrackActionBatchItem = SAMTrackActionInput['batch'][number];
type FrameObjectMap = Record<number, SerializedShape | SerializedTrack>;
type TrackShapeMap = Record<number, FrameObjectMap>;

function processShape(
    shape: SerializedShape,
    removeFrameIds: Array<number>,
    inTracksRecord: TrackShapeMap,
    inDependRecord: Array<FrameObjectMap>,
): void {
    if ('track_id' in shape) {
        const trackID = shape.track_id as number; // TODO: track_id attribute maybe support future
        const trackFrames = inTracksRecord[trackID] ?? {};

        if (trackFrames[shape.frame]) {
            throw new Error(
                `Invalid annotation: multiple objects with track_id=${trackID} found in frame=${shape.frame}. ` +
                'A track_id must be unique within a single frame.',
            );
        }

        if (!removeFrameIds.includes(shape.frame)) {
            trackFrames[shape.frame] = shape;
            inTracksRecord[trackID] = trackFrames;
        }
    } else if (!removeFrameIds.includes(shape.frame)) {
        inDependRecord.push({ [shape.frame]: shape });
    }
}

function processTrack(
    track: SerializedTrack,
    action: BaseSAMTrackAction,
    removeFrameIds: Array<number>,
    inTracksRecord: TrackShapeMap,
    inDependRecord: Array<FrameObjectMap>,
): void {
    if ('track_id' in track) {
        const trackID = track.track_id as number; // TODO: track_id attribute maybe support future
        const trackFrames = inTracksRecord[trackID] ?? {};

        for (let { frame } = track.shapes[0]; frame <= action.frameTo; frame++) {
            const keyframe = track.shapes.filter((kf) => kf.frame <= frame).at(-1);
            if (!keyframe || keyframe.outside) break;
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
            inTracksRecord[trackID] = trackFrames;
        }
    } else {
        const trackRecord: Record<number, SerializedTrack> = {};

        for (let { frame } = track.shapes[0]; frame <= action.frameTo; frame++) {
            const keyframe = track.shapes.filter((kf) => kf.frame <= frame).at(-1);
            if (!keyframe || keyframe.outside) break;
            if (!removeFrameIds.includes(frame)) {
                trackRecord[frame] = track;
            }
        }

        if (Object.keys(trackRecord).length > 0) {
            inDependRecord.push(trackRecord);
        }
    }
}

class SAMTrackObject {
    private order: Array<number>;
    private readonly removeFrameIds: Array<number>;

    private readonly type: string;
    private readonly labelId: number;
    private readonly objectId: string;
    private readonly nonCond: Array<number>;
    private readonly cond: Record<number, SerializedShape | SerializedTrack>;

    private readonly frameFrom: number;
    private readonly frameTo: number;

    private updateOrder(): void {
        const order: number[] = [];
        const frames: number[] = [
            ...Object.keys(this.cond).map(Number),
            ...this.nonCond,
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
            Using a simple non-cryptographic hash algorithm: djb2 xor variant.

            The initial value 5381 is commonly used by djb2.
            This implementation updates the hash as:

                hash = (hash * 33) ^ c

            where c is the current character code.
         */
        let hash = 5381;

        for (let i = 0; i < pointsCluster.length; i++) {
            hash = (hash * 33 + pointsCluster.charCodeAt(i)) % 0x100000000;
        }

        return hash.toString(16).padStart(8, '0');
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
        return (this.frameTo - this.frameFrom + 1) - this.removeFrameIds.length -
            Object.keys({ ...this.cond }).length - this.nonCond.length;
    }

    generate(): SAMTrackActionBatchItem | undefined {
        if (this.length === 0) return undefined;
        if (this.order.length === 0) {
            this.updateOrder();
            if (this.order.length === 0) return undefined;
        }

        return {
            frame: this.order.pop() as number,
            objectId: this.objectId,
            labelId: this.labelId,
            type: this.type,
            non_cond: this.nonCond,
            cond: Object.fromEntries(
                Object.entries(this.cond).map(([k, v]) => [k, {
                    id: Number(v.id),
                    type: 'shapes' in v ? 'track' : 'shape',
                }]),
            ),
        };
    }

    update(frame: number): void { this.nonCond.push(frame); }

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
        this.nonCond = [];
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
    const matchBatch = (samTrackObjects: Array<SAMTrackObject>, initialBatchSize: number): number[][] => {
        const result: number[][] = [];
        const trackFramesCount = samTrackObjects.map((object) => object.length);

        let batchSize = initialBatchSize;
        if (batchSize <= 0) {
            batchSize = trackFramesCount.length > 0 ? Math.max(...trackFramesCount) : 1;
        }

        while (trackFramesCount.some((count) => count > 0)) {
            const batch: number[] = [];
            for (let i = 0; i < trackFramesCount.length; i++) {
                if (trackFramesCount[i] > 0) {
                    batch.push(i);
                    trackFramesCount[i] -= 1;
                    if (batch.length === batchSize) {
                        break;
                    }
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
            Array.from({ length: action.frameTo - action.frameFrom + 1 }, (_, i) => action.frameFrom + i)
                .map(async (frame) => {
                    const frameData = await Object.getPrototypeOf(instance).frames
                        .get.implementation.call(instance, frame);
                    return frameData.deleted ? frame : null;
                }),
        )).filter((frame): frame is number => frame !== null);

        // Callback call must be late for action.init, because callback will get action attribute: frameFrom, frameTo
        const samTrackObjects: Array<SAMTrackObject> = await callback(removeFrameIds);
        if (samTrackObjects.length === 0) {
            await showMessageWithPause('No Annotations to process', 100, 1500);
            return;
        }

        // Iterate SAM Video Tracking
        await showMessageWithPause('Actions are running', 0, 500);
        const samTrackPredBatch = matchBatch(samTrackObjects, batchSize);
        const samTrackPredIters = samTrackPredBatch.length;
        for (let i = 0; i < samTrackPredIters; i++) {
            const samTrackObjectsBatch = samTrackPredBatch[i].map((gid) => samTrackObjects[gid]);

            const generatedBatch = samTrackObjectsBatch
                .map((object) => ({ object, item: object.generate() }))
                .filter((value): value is { object: SAMTrackObject; item: SAMTrackActionBatchItem } => (
                    typeof value.item !== 'undefined'
                ));

            if (generatedBatch.length === 0) {
                continue;
            }

            // NOTE: Maybe confidence need to save in future
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            for (const [bid, { frame, created, confidence }] of (await action.run({
                batch: generatedBatch.map(({ item }) => item),
                onProgress: decoratedOnProgress,
                cancelled,
            })).entries()) {
                generatedBatch[bid].object.update(frame); // update state

                if (created !== null) {
                    await instance.annotations.commit(
                        { shapes: [created as SerializedShape], tags: [], tracks: [] },
                        { shapes: [], tags: [], tracks: [] },
                        frame,
                    );
                }
            }

            const progress = Math.ceil(((i + 1) / samTrackPredIters) * 100);
            decoratedOnProgress('Actions are running', progress);
            if (cancelled()) {
                return;
            }
        }

        await showMessageWithPause('Committing handled objects', 100, 1500);
        if (cancelled()) {
            return;
        }
        await event.close();
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
            const inTracksRecord: Record<number, Record<number, SerializedShape | SerializedTrack>> = {};
            const inDependRecord: Array<Record<number, SerializedShape | SerializedTrack>> = [];

            const exportedCollection = getCollection(instance).export();
            validateClientIDs(exportedCollection);

            const filteredByAction = action.applyFilter({
                shapes: exportedCollection.shapes,
                tracks: exportedCollection.tracks,
            });

            const annotationsFilter = new AnnotationsFilter(null);
            const filteredClientIDs = annotationsFilter.filterSerializedCollection({
                tags: [],
                shapes: filteredByAction.shapes,
                tracks: filteredByAction.tracks,
            }, instance.labels, filters);

            for (const shape of filteredByAction.shapes) {
                if (!filteredClientIDs.shapes.includes(shape.clientID) || !shape.id) continue;
                processShape(shape, removeFrameIds, inTracksRecord, inDependRecord);
            }
            for (const track of filteredByAction.tracks) {
                if (!filteredClientIDs.tracks.includes(track.clientID) || !track.id) continue;
                processTrack(track, action, removeFrameIds, inTracksRecord, inDependRecord);
            }

            return [...inDependRecord, ...Object.values(inTracksRecord)].map(
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
            const inTracksRecord: Record<number, Record<number, SerializedShape | SerializedTrack>> = {};
            const inDependRecord: Array<Record<number, SerializedShape | SerializedTrack>> = [];
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
                processShape(shape, removeFrameIds, inTracksRecord, inDependRecord);
            }

            for (const track of filteredByAction.tracks) {
                processTrack(track, action, removeFrameIds, inTracksRecord, inDependRecord);
            }

            return [...inDependRecord, ...Object.values(inTracksRecord)].map(
                (record) => new SAMTrackObject(record, removeFrameIds, action.frameFrom, action.frameTo)).filter(
                (obj: SAMTrackObject): boolean => obj.length > 0,
            );
        },
        onProgress,
        cancelled,
    );
}
