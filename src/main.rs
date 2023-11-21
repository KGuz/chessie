use anyhow::Result;
use clap::Parser;

const ABOUT: &str = "Chessie (Chess Inference Engine) is a cli tool that aims to provide support 
for analyzing over the board chess games by translating photos/videos from actual chess positions 
into official chess notation using computer vision and machine learning algorithms.";

/// CLI tool to facilitate the analysis of over the board chess games
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = ABOUT)]
pub struct Args {
    /// Media file containing footage from chess game
    pub file: String,

    /// Generate official documentation (PGN or FEN)
    #[clap(short, long)]
    pub doc: bool,

    /// Generate visual representation (GIF or JPG)
    #[clap(short, long)]
    pub brd: bool,

    /// Generate neural network context view
    #[clap(short, long)]
    pub ctx: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let src = image::open(args.file)?.to_rgb8();

    let img = chessie::preprocess(&src);
    chessie::detect_board(&img)?;

    Ok(())
}
