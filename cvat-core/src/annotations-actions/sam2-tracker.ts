// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { SerializedCollection } from '../server-response-types';
import { BaseSAMTrackAction, SAMTrackActionInput, SAMTrackActionOutput } from './base-sam-track-actions';
import { ActionParameters, ActionParameterType } from './base-action';
import { Job } from '../session';
import { ShapeType, ModelKind, Source } from '../enums';
import ObjectState from '../object-state';
import LambdaManager from '../lambda-manager';
import MLModel from '../ml-model';

const MODEL_PARAMETER = 'Deployed Model';
const THRESHOLD_PARAMETER = 'Threshold';
const FRAME_RANGE_PARAMETER = 'Frame Range Setting Panel for Object Tracking';

export class SAM2Tracker extends BaseSAMTrackAction {
    private session: Job;
    private threshold: number;
    private model: MLModel;

    private static async getSAM2TrackerModels(): Promise<MLModel[]> {
        const { models } = await LambdaManager.list();

        return models.filter((model: MLModel) => (model.kind === ModelKind.VTRACKER));
    }

    private static async getDefaultModelName(): Promise<string> {
        const models = await SAM2Tracker.getSAM2TrackerModels();

        if (models.length === 0) {
            throw new Error('No vtracker model found');
        }

        return models[0].name;
    }

    public async init(sessionInstance: Job, parameters: Record<string, string | number>): Promise<void> {
        this.session = sessionInstance;
        let framesRange: number[] | null = null;
        // Extra parameters
        this.threshold = parameters[THRESHOLD_PARAMETER] as number;
        const modelName = parameters[MODEL_PARAMETER] as string;
        const models = await SAM2Tracker.getSAM2TrackerModels();
        const selectedModel = models.find((model: MLModel) => model.name === modelName);

        if (!selectedModel) {
            throw new Error(`vtracker model "${modelName}" is not available`);
        }

        this.model = selectedModel;
        const framesRangeString = parameters[FRAME_RANGE_PARAMETER];
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

        const payload = { jobId: this.session.id, threshold: this.threshold, batch };
        const response = await LambdaManager.call(this.session.taskId, this.model, payload);

        if (!Array.isArray(response)) {
            throw new Error(`vtracker model "${this.model.name}" returned invalid response`);
        }

        if (response.length !== batch.length) {
            throw new Error(
                `vtracker model "${this.model.name}" returned ${response.length} results for ${batch.length} items`,
            );
        }

        return response.map((result, idx) => {
            const item = batch[idx];

            if (result === null) {
                return { frame: item.frame, created: null, confidence: 0.0 };
            }

            return {
                frame: item.frame,
                confidence: result.confidence ?? 0.0,
                created: {
                    label_id: result.labelId ?? item.labelId,
                    frame: item.frame,
                    group: 0,
                    source: Source.SEMI_AUTO,
                    score: result.confidence ?? 0.0,
                    attributes: [],
                    elements: [],
                    occluded: false,
                    outside: false,
                    points: result.points,
                    rotation: 0,
                    z_order: 0,
                    type: result.type,
                },
            };
        });
    }

    public applyFilter(
        input: Pick<SerializedCollection, 'shapes' | 'tracks'>,
    ): Pick<SerializedCollection, 'shapes' | 'tracks'> {
        const { shapes, tracks } = input;
        const targetShapesType = [ShapeType.POLYGON, ShapeType.MASK];
        return {
            shapes: shapes.filter((shape) => targetShapesType.includes(shape.type)),
            tracks: tracks.filter((track) => (
                track.shapes.length > 0 && targetShapesType.includes(track.shapes[0].type)
            )),
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
            [THRESHOLD_PARAMETER]: {
                type: ActionParameterType.NUMBER,
                values: ['0', '1', '0.01'],
                defaultValue: '0.6',
            },
            [MODEL_PARAMETER]: {
                type: ActionParameterType.SELECT,
                values: async () => (await SAM2Tracker.getSAM2TrackerModels()).map((model: MLModel) => model.name),
                defaultValue: SAM2Tracker.getDefaultModelName,
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
            [FRAME_RANGE_PARAMETER]: {
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
