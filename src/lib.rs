//! Kerr Engine — wave-native compute for Kerr-ODE architecture.
//!
//! Library interface exposing the public API for external crates.

pub mod model;
pub mod backend;
pub mod bpe;
pub mod checkpoint;
pub mod data;
pub mod gpu_buffers;
pub mod gpu_ops;
pub mod init;
pub mod optim;
pub mod rng;
pub mod backward;
pub mod pipeline;
pub mod gpu;
pub mod gpu_backend;
pub mod gpu_dispatch;
pub mod gpu_pipelines;
pub mod gpu_persistent;
pub mod gpu_validate;
pub mod grad_test;
pub mod train;
pub mod weights;
