// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { BaseSAMTrackAction, SAMTrackActionInput, SAMTrackActionOutput } from './base-sam-track-actions';
import { ActionParameters, ActionParameterType } from './base-action';
import { Job, Task } from '../session';
import { ShapeType } from '../enums';
import ObjectState from '../object-state';

export class SAM2Tracker extends BaseSAMTrackAction {
    private session: Job | Task;
    private convertPolygonShapesToTracks: boolean;

    public async init(sessionInstance: Job | Task, parameters: Record<string, string | number>): Promise<void> {
        this.session = sessionInstance;
        let framesRange: number[] | null = null;
        const framesRangeString = parameters['Frame Range Setting Panel for Object Tracking'];
        if (typeof framesRangeString === 'string' && framesRangeString.trim() !== '') {
            const parsed = framesRangeString.split('-').map((v) => v.trim()).map(Number);
            if (parsed.length === 2 && parsed.every((n) => !Number.isNaN(n))) framesRange = parsed;
        }
        [this.frameFrom, this.frameTo] = framesRange ?? [sessionInstance.startFrame, sessionInstance.stopFrame];
        this.convertPolygonShapesToTracks = parameters['Convert polygon shapes to tracks'] === 'true';
    }

    public async destroy(): Promise<void> {
        // nothing to destroy
    }

    public async run(input: SAMTrackActionInput): Promise<SAMTrackActionOutput> {
        const { collection, onProgress, cancelled, frameData: { number, width, height } } = input;

        if (collection.shapes.length === 0 && collection.tracks.length === 0) {
            throw new Error('The current job must have at least one polygon or mask annotations');
        }

        // TODO: call model api to auto segment annotation

        return {
            created: { shapes: [], tracks: [] },
            deleted: { shapes: [], tracks: [] },
        };
    }

    public applyFilter(
        input: Pick<SAMTrackActionInput, 'collection' | 'frameData'>,
    ): SAMTrackActionInput['collection'] {
        const { collection } = input;
        const targetShapesType = [ShapeType.POLYGON, ShapeType.MASK];
        return {
            shapes: collection.shapes
                .filter((shape) => targetShapesType.includes(shape.type)),
            tracks: collection.tracks
                .filter((track) => targetShapesType.includes(track.shapes[0].type)),
        };
    }

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    public isApplicableForObject(objectState: ObjectState): boolean {
        return false;
    }

    public get name(): string {
        return 'Segment Anything 2: Tracker (All Frames)';
    }

    public get parameters(): ActionParameters | null {
        return {
            'Convert polygon shapes to tracks': {
                type: ActionParameterType.CHECKBOX,
                values: ['true', 'false'],
                defaultValue: 'true',
            },
        };
    }
}

export class SAM2TrackerObject extends SAM2Tracker {
    public isApplicableForObject(objectState: ObjectState): boolean {
        return [ShapeType.POLYGON, ShapeType.MASK].includes(objectState.shapeType);
    }

    public isApplicableForObjectOnly(): boolean {
        return true; // Action only for object
    }

    public get name(): string {
        return 'Segment Anything 2: Tracker (Object)';
    }

    public get parameters(): ActionParameters | null {
        return {
            ...super.parameters,
            'Frame Range Setting Panel for Object Tracking': {
                type: ActionParameterType.FRAMESRANGESELECTOR,
                values: ({ instance }) => {
                    if (instance instanceof Job) {
                        return [instance.startFrame, instance.stopFrame].map((val) => val.toString());
                    }
                    return [0, instance.size - 1].map((val) => val.toString());
                },
                defaultValue: ({ instance }) => {
                    if (instance instanceof Job) {
                        return [instance.startFrame, instance.stopFrame].map((val) => val.toString()).join('-');
                    }
                    return [0, instance.size - 1].map((val) => val.toString()).join('-');
                },
            },
        };
    }
}
