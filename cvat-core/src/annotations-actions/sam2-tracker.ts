// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { SerializedCollection } from '../server-response-types';
import { BaseSAMTrackAction, SAMTrackActionInput, SAMTrackActionOutput } from './base-sam-track-actions';
import { ActionParameters, ActionParameterType } from './base-action';
import { Job, Task } from '../session';
import { ShapeType } from '../enums';
import ObjectState from '../object-state';
import serverProxy from "../server-proxy";

export class SAM2Tracker extends BaseSAMTrackAction {
    private session: Job;
    private threshold: number;

    public async init(sessionInstance: Job, parameters: Record<string, string | number>): Promise<void> {
        this.session = sessionInstance;
        let framesRange: number[] | null = null;
        // Extra parameters
        this.threshold = parameters['Threshold'] as number;
        const framesRangeString = parameters['Frame Range Setting Panel for Object Tracking'];
        if (typeof framesRangeString === 'string' && framesRangeString.trim() !== '') {
            const parsed = framesRangeString.split('-').map((v) => v.trim()).map(Number);
            if (parsed.length === 2 && parsed.every((n) => !Number.isNaN(n))) framesRange = parsed;
        }
        [this.frameFrom, this.frameTo] = framesRange ?? [sessionInstance.startFrame, sessionInstance.stopFrame];
    }

    public async destroy(): Promise<void> {
        // nothing to destroy
    }

    public async run(input: SAMTrackActionInput): Promise<SAMTrackActionOutput> {
        const { batch } = input;

        const payload = {jobId: this.session.id, threshold: this.threshold, batch: batch};

        // TODO: call model api to auto segment annotation
        // const response = await serverProxy.lambda.call("xxx", payload);

        for (const item of batch) {
            console.log(`
                frame: ${String(item.frame).padStart(4)}
                objectId: ${item.objectId}
                labelId: ${item.labelId}
                type: ${item.type}
                cond: ${Object.keys(item.cond)}
                non-cond: ${Object.keys(item.non_cond)}
            `);
        }
        console.log("===========");
        console.dir(payload);
        // console.log(JSON.stringify(payload));

        return batch.map(item => ({ frame: item.frame, created: null, confidence: 0.0 }));
    }

    public applyFilter(
        input: Pick<SerializedCollection, 'shapes' | 'tracks'>,
    ): Pick<SerializedCollection, 'shapes' | 'tracks'> {
        const { shapes, tracks } = input;
        const targetShapesType = [ShapeType.POLYGON, ShapeType.MASK];
        return {
            shapes: shapes.filter((shape) => targetShapesType.includes(shape.type)),
            tracks: tracks.filter((track) => targetShapesType.includes(track.shapes[0].type)),
        };
    }

    public isApplicableForObject(objectState: ObjectState): boolean {
        return false;
    }

    public get name(): string {
        return 'Segment Anything 2: Tracker (All Frames)';
    }

    public get parameters(): ActionParameters | null {
        return {
            "Threshold": {
                type: ActionParameterType.NUMBER,
                values: ['0', '1', '0.01'],
                defaultValue: '0.6',
            }
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
