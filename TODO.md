# TODO: Improve Video Detection Accuracy for Slow Movements

## Step 1: Enhance Frame Sampling
- [x] Increase sample_rate to capture more frames (e.g., up to 100 frames) for better motion analysis.

## Step 2: Refine Motion Detection Thresholds
- [x] Lower the threshold for low motion detection (e.g., from 0.5 to 0.3 for high score).
- [x] Increase the score weight for low motion (e.g., from 30 to 50 points).

## Step 3: Add Advanced Motion Metrics
- [x] Calculate variance in motion values to detect inconsistent slow movements.
- [x] Add scoring for high variance in slow motion, indicating potential fakeness.

## Step 4: Update Scoring Logic
- [x] Adjust overall scoring to prioritize slow movement indicators.
- [ ] Ensure verdict thresholds remain balanced.

## Step 5: Test and Validate
- [ ] Run tests on sample videos to verify improved accuracy for slow movements.
- [ ] Update any dependent files if needed.
