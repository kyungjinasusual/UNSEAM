# Phase 3: Deep Learning Methods

Neural network-based approaches for event segmentation and causal inference.

## Planned Implementations

### GST-UNet (gst_unet/)
Spatiotemporal causal inference framework (Oprescu et al.)
- U-Net encoder for spatiotemporal features
- ConvLSTM for temporal dynamics
- G-Heads for causal inference
- Counterfactual simulation capabilities

### Neural Sequence Models (neural_seq/)
State-of-the-art sequence models for fMRI analysis:
- **SWIFT**: Efficient transformer for brain state dynamics
- **NeuroMamba**: State-space model for neural sequences
- Other emerging architectures

## Status
Planning phase - implementations coming after HMM/BSDS validation

## Dependencies (planned)
```
torch>=2.0
einops
mamba-ssm  # for NeuroMamba
```
