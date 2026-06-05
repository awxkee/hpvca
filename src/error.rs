use thiserror::Error;

#[derive(Error, Debug)]
pub enum EncodeError {
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("Invalid input: plane sizes or parameters are inconsistent")]
    InvalidInput,

    #[error("DCT block error: {0}")]
    DctError(String),

    #[error("Bitstream write error: {0}")]
    BitstreamError(String),

    #[error("ISOBMFF error: {0}")]
    IsobmffError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
