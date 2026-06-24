// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect } from 'react';
import { ArrowLeftOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { Button } from 'antd';
import InputNumber from 'antd/lib/input-number';
import Slider from 'antd/lib/slider';

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
