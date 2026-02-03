"""
Main Application Entry Point
----------------------------
Smart Text Assistant - Professional AI Writing Tool

Usage:
    python app.py
    
Or with custom parameters:
    python app.py --corpus path/to/corpus.txt --keyboard path/to/keyboard.txt

Authors: El Guelta Mohamad Saber, El Hadifi Soukaina
"""

import argparse
import os
import sys
from text_processor_enhanced import HybridTextProcessor
from app_simple import SmartTextAssistant


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Smart Text Assistant - Professional AI Writing Tool'
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        default='big_data.txt',
        help='Path to text corpus file (default: big_data.txt)'
    )
    
    parser.add_argument(
        '--keyboard',
        type=str,
        default='qwerty_graph.txt',
        help='Path to keyboard graph file (default: qwerty_graph.txt)'
    )
    
    parser.add_argument(
        '--ngram-size',
        type=int,
        default=2,
        help='Size of n-grams (default: 2)'
    )
    
    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.1,
        help='Smoothing parameter for n-gram model (default: 0.1)'
    )
    
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum word frequency for dictionary (default: 2)'
    )
    
    parser.add_argument(
        '--no-transformers',
        action='store_true',
        help='Disable transformer models (BERT & GPT2) - use only n-gram'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    
    parser.add_argument(
        '--server-name',
        type=str,
        default='0.0.0.0',
        help='Server name (default: 0.0.0.0 for Docker)'
    )
    
    parser.add_argument(
        '--server-port',
        type=int,
        default=7860,
        help='Server port (default: 7860)'
    )
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    print("=" * 60)
    print("✨ SMART TEXT ASSISTANT")
    print("   Professional AI-Powered Writing Tool")
    print("=" * 60)
    print()
    
    # Parse arguments
    args = parse_args()
    
    # Verify files exist
    if not os.path.exists(args.corpus):
        print(f"❌ Error: Corpus file not found: {args.corpus}")
        print("   Please provide a valid corpus file path.")
        sys.exit(1)
    
    if not os.path.exists(args.keyboard):
        print(f"❌ Error: Keyboard graph file not found: {args.keyboard}")
        print("   Please provide a valid keyboard graph file path.")
        sys.exit(1)
    
    # Display configuration
    print("⚙️  Configuration:")
    print(f"   Corpus: {args.corpus}")
    print(f"   Keyboard Graph: {args.keyboard}")
    print(f"   N-gram Size: {args.ngram_size}")
    print(f"   Smoothing: {args.smoothing}")
    print(f"   Min Frequency: {args.min_frequency}")
    print(f"   Transformers: {'❌ Disabled' if args.no_transformers else '✅ Enabled'}")
    print()
    
    try:
        # Initialize the processor
        print("🚀 Initializing AI models...")
        print("   This may take a few minutes on first run...")
        print()
        
        processor = HybridTextProcessor(
            corpus_path=args.corpus,
            keyboard_graph_path=args.keyboard,
            ngram_size=args.ngram_size,
            smoothing_k=args.smoothing,
            min_frequency=args.min_frequency,
            use_transformers=not args.no_transformers
        )
        
        print()
        print("✅ AI models ready!")
        print()
        
        # Create and launch the app
        print("🌐 Launching Smart Text Assistant...")
        app = SmartTextAssistant(processor)
        
        # Launch with specified settings
        app.launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
