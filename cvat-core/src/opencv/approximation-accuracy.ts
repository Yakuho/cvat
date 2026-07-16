// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

export const MIN_APPROXIMATION_ACCURACY = 0;
export const MAX_APPROXIMATION_ACCURACY = 13;
export const DEFAULT_APPROXIMATION_ACCURACY = 9;
export const APPROXIMATION_ACCURACY_STEP = 1;

export function thresholdFromAccuracy(accuracy: number): number {
    // Convert accuracy (0-13 scale) to epsilon threshold
    // This matches the approximation accuracy slider
    const approxPolyMaxDistance = MAX_APPROXIMATION_ACCURACY - accuracy;

    if (approxPolyMaxDistance > 0) {
        if (approxPolyMaxDistance <= 8) {
            // Linear interpolation from (1, 0.25) to (8, 3)
            return (2.75 * approxPolyMaxDistance - 1) / 7;
        }
        // Exponential: 4 for 9, 8 for 10, 16 for 11, 32 for 12, 64 for 13
        return 2 ** (approxPolyMaxDistance - 7);
    }
    return 0;
}
