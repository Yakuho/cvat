// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect } from 'react';
import { ArrowLeftOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { Button } from 'antd';
import InputNumber from 'antd/lib/input-number';
import Slider from 'antd/lib/slider';

import {
    APPROXIMATION_ACCURACY_STEP,
    DEFAULT_APPROXIMATION_ACCURACY,
    MAX_APPROXIMATION_ACCURACY,
    MIN_APPROXIMATION_ACCURACY,
    thresholdFromAccuracy,
} from 'cvat-core-wrapper';
import openCVWrapper from 'utils/opencv-wrapper/opencv-wrapper';

const EXAMPLE_SHAPES: [number, number][] = [
    [28, 105], [31, 93], [36, 82], [43, 72], [51, 65], [60, 60], [70, 57], [80, 56],
    [90, 57], [100, 60], [110, 64], [120, 68], [130, 69], [140, 67], [149, 61], [157, 52],
    [165, 39], [173, 28], [180, 24], [187, 29], [194, 43], [201, 61], [208, 77], [216, 88],
    [225, 94], [235, 96], [245, 94], [255, 89], [265, 80], [274, 71], [282, 68], [289, 73],
    [294, 84], [296, 98], [294, 111], [288, 122], [279, 130], [268, 135], [256, 137], [244, 136],
    [232, 132], [220, 126], [208, 121], [196, 118], [184, 117], [172, 119], [160, 124], [148, 131],
    [136, 138], [124, 141], [112, 138], [101, 131], [91, 122], [81, 114], [71, 109], [61, 108],
    [51, 112], [42, 117], [34, 116], [29, 111],
];

function accuracyFromThreshold(threshold: number): number {
    if (!Number.isFinite(threshold)) {
        return DEFAULT_APPROXIMATION_ACCURACY;
    }

    let closestAccuracy = DEFAULT_APPROXIMATION_ACCURACY;
    let closestDistance = Number.POSITIVE_INFINITY;
    for (let accuracy = MIN_APPROXIMATION_ACCURACY; accuracy <= MAX_APPROXIMATION_ACCURACY; accuracy++) {
        const distance = Math.abs(thresholdFromAccuracy(accuracy) - threshold);
        if (distance < closestDistance) {
            closestAccuracy = accuracy;
            closestDistance = distance;
        }
    }

    return closestAccuracy;
}

interface ApproxThresholdSelectorProps {
    value: string;
    onChange: (value: string) => void;
    controlPointsSize: number;
}

export function ApproxThresholdSelector({
    value, onChange, controlPointsSize,
}: ApproxThresholdSelectorProps): JSX.Element {
    const [accuracy, setAccuracy] = useState(() => accuracyFromThreshold(Number(value)));
    const [previewPoints, setPreviewPoints] = useState<[number, number][]>([]);
    const [previewLoading, setPreviewLoading] = useState(true);
    const [previewAvailable, setPreviewAvailable] = useState(true);

    useEffect(() => {
        setAccuracy(accuracyFromThreshold(Number(value)));
    }, [value]);

    useEffect(() => {
        let cancelled = false;
        setPreviewLoading(true);

        const updatePreview = async (): Promise<void> => {
            try {
                if (!openCVWrapper.isInitialized) {
                    await openCVWrapper.initialize(() => {});
                }

                const points = openCVWrapper.contours.approxPoly(
                    EXAMPLE_SHAPES,
                    thresholdFromAccuracy(accuracy),
                    true,
                );
                if (!cancelled) {
                    setPreviewPoints(points);
                    setPreviewAvailable(true);
                }
            } catch (_error) {
                if (!cancelled) {
                    setPreviewAvailable(false);
                }
            } finally {
                if (!cancelled) {
                    setPreviewLoading(false);
                }
            }
        };

        void updatePreview();

        return () => { cancelled = true; };
    }, [accuracy]);

    const originalPolygon = EXAMPLE_SHAPES.map((point) => point.join(',')).join(' ');
    const simplifiedPolygon = previewPoints.map((point) => point.join(',')).join(' ');

    return (
        <div className='cvat-approx-threshold-selector'>
            <div className='cvat-approx-threshold-preview'>
                {previewLoading && <span className='cvat-approx-threshold-preview-state'>Loading preview...</span>}
                {!previewLoading && !previewAvailable && (
                    <span className='cvat-approx-threshold-preview-state'>Preview unavailable</span>
                )}
                {!previewLoading && previewAvailable && (
                    <svg viewBox='0 0 324 164' role='img' aria-label='Polygon approximation preview'>
                        <polygon className='cvat-approx-threshold-original-contour' points={originalPolygon} />
                        <polygon className='cvat-approx-threshold-simplified-contour' points={simplifiedPolygon} />
                        {previewPoints.map(([x, y], index) => (
                            <circle
                                key={index}
                                cx={x}
                                cy={y}
                                r={controlPointsSize}
                                fill='white'
                                stroke='black'
                            />
                        ))}
                    </svg>
                )}
            </div>
            <div className='cvat-approx-threshold-summary'>
                <span>Example points</span>
                <span className='cvat-approx-threshold-point-count'>
                    {previewAvailable && !previewLoading ? `${previewPoints.length} points` : '— points'}
                </span>
            </div>
            <Slider
                min={MIN_APPROXIMATION_ACCURACY}
                max={MAX_APPROXIMATION_ACCURACY}
                step={APPROXIMATION_ACCURACY_STEP}
                value={accuracy}
                dots
                tooltip={{ open: false }}
                onChange={(updatedAccuracy: number) => {
                    setAccuracy(updatedAccuracy);
                    onChange(String(thresholdFromAccuracy(updatedAccuracy)));
                }}
            />
        </div>
    );
}

interface FramesRangeSelectorProps {
    value: string;
    onChange: (value: string) => void;
    startFrame: number;
    stopFrame: number;
    frameNumber: number;
}

export function FramesRangeSelector({
    value, onChange, startFrame, stopFrame, frameNumber,
}: FramesRangeSelectorProps): JSX.Element {
    const [frameRange, setFrameRange] = useState<[number, number]>([startFrame, stopFrame]);

    const normalizeFrameRange = ([v1, v2]: number[]): [number, number] => {
        let lo = Math.min(v1, v2);
        let hi = Math.max(v1, v2);

        if (hi < frameNumber) hi = frameNumber;
        if (lo > frameNumber) lo = frameNumber;

        if (hi === lo) {
            if (lo > startFrame) lo -= 1;
            else hi += 1;
        }

        lo = Math.max(lo, startFrame);
        hi = Math.min(hi, stopFrame);

        return [lo, hi];
    };

    const formatFrameRange = (range: [number, number]): string => `${range[0]}-${range[1]}`;

    const updateFrameRange = (range: number[]): void => {
        const normalizedRange = normalizeFrameRange(range);
        setFrameRange(normalizedRange);
        onChange(formatFrameRange(normalizedRange));
    };

    useEffect(() => {
        const parts = value.split('-').map((v) => {
            const num = Number(v.trim());
            return !Number.isNaN(num) ? num : null;
        }).filter((v): v is number => v !== null);

        if (parts.length === 2) {
            const normalizedRange = normalizeFrameRange([
                Math.max(Math.min(parts[0], parts[1]), startFrame),
                Math.min(Math.max(parts[0], parts[1]), stopFrame),
            ]);

            setFrameRange(normalizedRange);
            if (formatFrameRange(normalizedRange) !== value) {
                onChange(formatFrameRange(normalizedRange));
            }
        }
    }, [value, startFrame, stopFrame, frameNumber]);

    const trackFrames = Math.abs(frameRange[1] - frameRange[0]);
    const backwardInputMin = frameRange[1] === frameNumber ? 1 : 0;
    const forwardInputMin = frameRange[0] === frameNumber ? 1 : 0;
    const backwardTrackAvailable = frameNumber > startFrame;
    const forwardTrackAvailable = frameNumber < stopFrame;
    const backwardActive = frameRange[0] < frameNumber;
    const forwardActive = frameRange[1] > frameNumber;

    return (
        <div className='cvat-frames-range-selector'>
            <div className='cvat-frames-range-selector-header'>
                <span className='cvat-frames-range-selector-count'>
                    {trackFrames}
                </span>
                <span className='cvat-frames-range-selector-text'>
                    frames will be track.
                </span>
            </div>

            <div className='cvat-frames-range-selector-direction'>
                <span className='label'>Please, specify a direction for tracking: </span>

                <div className='direction-buttons'>
                    <Button
                        size='middle'
                        type={backwardActive ? 'primary' : 'default'}
                        disabled={!backwardTrackAvailable}
                        onClick={() => {
                            if (backwardActive) {
                                const isMinBackward = frameNumber - frameRange[0] === 1 &&
                                    frameRange[1] === frameNumber;
                                if (isMinBackward) {
                                    updateFrameRange([startFrame, frameRange[1]]);
                                } else {
                                    const newLo = frameRange[1] === frameNumber ? frameNumber - 1 : frameNumber;
                                    updateFrameRange([newLo, frameRange[1]]);
                                }
                            } else {
                                updateFrameRange([startFrame, frameRange[1]]);
                            }
                        }}
                    >
                        <ArrowLeftOutlined />
                        backward
                    </Button>

                    <Button
                        size='middle'
                        type={forwardActive ? 'primary' : 'default'}
                        disabled={!forwardTrackAvailable}
                        onClick={() => {
                            if (forwardActive) {
                                const isMinForward = frameRange[1] - frameNumber === 1 &&
                                    frameRange[0] === frameNumber;
                                if (isMinForward) {
                                    updateFrameRange([frameRange[0], stopFrame]);
                                } else {
                                    const newHi = frameRange[0] === frameNumber ? frameNumber + 1 : frameNumber;
                                    updateFrameRange([frameRange[0], newHi]);
                                }
                            } else {
                                updateFrameRange([frameRange[0], stopFrame]);
                            }
                        }}
                    >
                        forward
                        <ArrowRightOutlined />
                    </Button>
                </div>
            </div>

            <div className='cvat-frames-range-selector-range'>
                <div className='label'>Or specify a range where track will be: </div>

                <div className='range-wrapper'>
                    <InputNumber
                        size='middle'
                        min={backwardInputMin}
                        max={frameNumber - startFrame}
                        value={frameNumber - frameRange[0]}
                        disabled={!backwardTrackAvailable}
                        onChange={(v: number | null) => {
                            if (typeof v === 'number') {
                                updateFrameRange([frameNumber - v, frameRange[1]]);
                            }
                        }}
                    />

                    <Slider
                        range
                        min={startFrame}
                        max={stopFrame}
                        value={[frameRange[0], frameRange[1]]}
                        onChange={updateFrameRange}
                    />

                    <InputNumber
                        size='middle'
                        min={forwardInputMin}
                        max={stopFrame - frameNumber}
                        value={frameRange[1] - frameNumber}
                        disabled={!forwardTrackAvailable}
                        onChange={(v: number | null) => {
                            if (typeof v === 'number') {
                                updateFrameRange([frameRange[0], frameNumber + v]);
                            }
                        }}
                    />
                </div>
            </div>
        </div>
    );
}
