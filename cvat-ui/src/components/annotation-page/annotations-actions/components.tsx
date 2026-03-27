// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect } from 'react';
import { ArrowLeftOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { Button } from 'antd';
import InputNumber from 'antd/lib/input-number';
import Slider from 'antd/lib/slider';

export function FramesRangeSelector({
    value, onChange, startFrame, stopFrame, frameNumber,
}: {
    value: string, onChange: (val: string) => void, startFrame: number, stopFrame: number, frameNumber: number
}): JSX.Element {
    const [frameRange, setFrameRange] = useState<[number, number]>([startFrame, stopFrame]);

    const updateFrameRange = ([v1, v2]: number[]): void => {
        let newRange: [number, number];
        if (v1 < frameNumber && v2 < frameNumber) {
            newRange = [Math.min(v1, v2), frameNumber];
        } else if (v1 > frameNumber && v2 > frameNumber) {
            newRange = [frameNumber, Math.max(v1, v2)];
        } else if (v1 === v2) {
            if (v1 === startFrame) {
                newRange = [v1, frameNumber === startFrame ? startFrame + 1 : frameNumber];
            } else if (v2 === stopFrame) {
                newRange = [frameNumber === stopFrame ? stopFrame - 1 : frameNumber, v2];
            } else newRange = [v1, v1 + 1];
        } else {
            newRange = [Math.min(v1, v2), Math.max(v1, v2)];
        }

        setFrameRange(newRange);
        onChange(`${newRange[0]}-${newRange[1]}`);
    };

    useEffect(() => {
        const parts = value.split('-').map((v) => {
            const num = Number(v.trim());
            return !Number.isNaN(num) ? num : null;
        }).filter((v): v is number => v !== null);

        if (parts.length === 2) {
            setFrameRange([
                Math.max(Math.min(parts[0], parts[1]), startFrame),
                Math.min(Math.max(parts[0], parts[1]), stopFrame),
            ]);
        }
    }, [value, startFrame, stopFrame]);

    const trackFrames = Math.abs(frameRange[1] - frameRange[0]);
    const backwardTrackAvailable = frameNumber > startFrame;
    const forwardTrackAvailable = frameNumber < stopFrame;
    const backwardActive = frameRange[0] < frameNumber - (frameNumber === stopFrame ? 1 : 0);
    const forwardActive = frameRange[1] > frameNumber + (frameNumber === startFrame ? 1 : 0);

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
                                updateFrameRange([frameNumber, frameRange[1]]);
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
                                updateFrameRange([frameRange[0], frameNumber]);
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
                        min={backwardTrackAvailable && frameRange[1] <= frameNumber ? 1 : 0}
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
                        min={forwardTrackAvailable && frameRange[0] >= frameNumber ? 1 : 0}
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
