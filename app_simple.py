"""
Smart Text Assistant - Simple & Professional
-------------------------------------------
Auto-correction, Smart completion, and Guided text generation

Authors: El Guelta Mohamad Saber, El Hadifi Soukaina
"""

import gradio as gr
from text_processor_enhanced import HybridTextProcessor
import os


class SmartTextAssistant:
    """Professional minimalist text assistant."""
    
    def __init__(self, processor: HybridTextProcessor):
        self.processor = processor
        self.demo = self._create_interface()
    
    def auto_correct(self, text: str) -> str:
        """Automatically correct errors in text."""
        if not text or not text.strip():
            return text
        
        result = self.processor.correct_text(text, use_bert=True)
        return result['corrected']
    
    def smart_complete(self, text: str, num_words: int = 10) -> str:
        """Complete text intelligently."""
        if not text or not text.strip():
            return text
        
        # First auto-correct
        corrected = self.auto_correct(text)
        
        # Then complete
        result = self.processor.autocomplete_hybrid(corrected, num_words, use_gpt2=True)
        return result['recommended']
    
    def get_next_word_options(self, text: str, num_options: int = 5) -> list:
        """Get word suggestions for guided generation."""
        if not text or not text.strip():
            return ["the", "a", "in", "to", "and"]
        
        # Auto-correct first
        corrected = self.auto_correct(text)
        
        # Get predictions from both models
        options = []
        
        # Get n-gram suggestions
        tokens = self.processor._preprocess_text(corrected)
        if tokens:
            context = tuple(tokens[-(self.processor.ngram_size - 1):])
            predictions = self.processor.ngram_counts.get(context, {})
            if predictions:
                sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                options.extend([word for word, _ in sorted_preds[:num_options]])
        
        # Get GPT2 suggestions if available
        if self.processor.use_transformers and self.processor.transformer_models:
            try:
                bert_predictions = self.processor.transformer_models.bert_predict_masked_word(
                    corrected + " [MASK]", 
                    top_k=num_options
                )
                for word, _ in bert_predictions:
                    if word not in options:
                        options.append(word)
            except:
                pass
        
        # Return top options
        return options[:num_options] if options else ["the", "a", "is", "to", "and"]
    
    def add_word_to_text(self, current_text: str, word: str) -> tuple:
        """Add selected word to text and get new suggestions."""
        new_text = current_text.strip() + " " + word if current_text.strip() else word
        suggestions = self.get_next_word_options(new_text)
        return new_text, *suggestions
    
    def _create_interface(self):
        """Create minimalist professional interface."""
        
        # Custom CSS for clean, professional look
        custom_css = """
        .gradio-container {
            max-width: 900px !important;
            margin: auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .input-box textarea {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        .suggestion-btn {
            margin: 5px;
            padding: 10px 20px;
            font-size: 14px;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Smart Text Assistant", theme=gr.themes.Soft()) as demo:
            
            # Header
            gr.HTML("""
                <div class="main-header">
                    <h1 style="margin: 0; font-size: 32px;">✨ Smart Text Assistant</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.9;">Write better with AI-powered correction & completion</p>
                </div>
            """)
            
            # Main Tab: Smart Writing
            with gr.Tab("✍️ Smart Writing"):
                gr.Markdown("### Type or paste your text - errors are corrected automatically")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        input_text = gr.Textbox(
                            label="Your Text",
                            placeholder="Start typing here... Errors will be corrected automatically.",
                            lines=10,
                            elem_classes="input-box"
                        )
                    
                    with gr.Column(scale=2):
                        output_text = gr.Textbox(
                            label="Corrected Text",
                            lines=10,
                            elem_classes="input-box"
                        )
                
                with gr.Row():
                    correct_btn = gr.Button("🔍 Check & Correct", variant="secondary", size="sm")
                    complete_btn = gr.Button("🚀 Complete Text (10 words)", variant="primary", size="sm")
                
                # Auto-correct on change
                input_text.change(
                    fn=self.auto_correct,
                    inputs=input_text,
                    outputs=output_text
                )
                
                correct_btn.click(
                    fn=self.auto_correct,
                    inputs=input_text,
                    outputs=output_text
                )
                
                complete_btn.click(
                    fn=self.smart_complete,
                    inputs=input_text,
                    outputs=output_text
                )
            
            # Guided Writing Tab
            with gr.Tab("🎯 Guided Writing"):
                gr.Markdown("### Build your text word by word with AI suggestions")
                
                guided_text = gr.Textbox(
                    label="Your Text",
                    placeholder="Click a word suggestion below to start...",
                    lines=8
                )
                
                gr.Markdown("**💡 Choose the next word:**")
                
                with gr.Row():
                    word_btn_1 = gr.Button("", variant="secondary", elem_classes="suggestion-btn")
                    word_btn_2 = gr.Button("", variant="secondary", elem_classes="suggestion-btn")
                    word_btn_3 = gr.Button("", variant="secondary", elem_classes="suggestion-btn")
                    word_btn_4 = gr.Button("", variant="secondary", elem_classes="suggestion-btn")
                    word_btn_5 = gr.Button("", variant="secondary", elem_classes="suggestion-btn")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Suggestions", size="sm")
                    clear_btn = gr.Button("🗑️ Clear Text", size="sm")
                
                # Function to update buttons
                def update_suggestions(text):
                    suggestions = self.get_next_word_options(text, 5)
                    return [
                        gr.Button(suggestions[0] if len(suggestions) > 0 else "the"),
                        gr.Button(suggestions[1] if len(suggestions) > 1 else "a"),
                        gr.Button(suggestions[2] if len(suggestions) > 2 else "is"),
                        gr.Button(suggestions[3] if len(suggestions) > 3 else "to"),
                        gr.Button(suggestions[4] if len(suggestions) > 4 else "and"),
                    ]
                
                # Click handlers for word buttons
                def add_word_1(text):
                    suggestions = self.get_next_word_options(text, 5)
                    word = suggestions[0] if suggestions else "the"
                    new_text = f"{text} {word}".strip()
                    new_suggestions = self.get_next_word_options(new_text, 5)
                    return [
                        new_text,
                        gr.Button(new_suggestions[0] if len(new_suggestions) > 0 else "the"),
                        gr.Button(new_suggestions[1] if len(new_suggestions) > 1 else "a"),
                        gr.Button(new_suggestions[2] if len(new_suggestions) > 2 else "is"),
                        gr.Button(new_suggestions[3] if len(new_suggestions) > 3 else "to"),
                        gr.Button(new_suggestions[4] if len(new_suggestions) > 4 else "and"),
                    ]
                
                def add_word_2(text):
                    suggestions = self.get_next_word_options(text, 5)
                    word = suggestions[1] if len(suggestions) > 1 else "a"
                    new_text = f"{text} {word}".strip()
                    new_suggestions = self.get_next_word_options(new_text, 5)
                    return [
                        new_text,
                        gr.Button(new_suggestions[0] if len(new_suggestions) > 0 else "the"),
                        gr.Button(new_suggestions[1] if len(new_suggestions) > 1 else "a"),
                        gr.Button(new_suggestions[2] if len(new_suggestions) > 2 else "is"),
                        gr.Button(new_suggestions[3] if len(new_suggestions) > 3 else "to"),
                        gr.Button(new_suggestions[4] if len(new_suggestions) > 4 else "and"),
                    ]
                
                def add_word_3(text):
                    suggestions = self.get_next_word_options(text, 5)
                    word = suggestions[2] if len(suggestions) > 2 else "is"
                    new_text = f"{text} {word}".strip()
                    new_suggestions = self.get_next_word_options(new_text, 5)
                    return [
                        new_text,
                        gr.Button(new_suggestions[0] if len(new_suggestions) > 0 else "the"),
                        gr.Button(new_suggestions[1] if len(new_suggestions) > 1 else "a"),
                        gr.Button(new_suggestions[2] if len(new_suggestions) > 2 else "is"),
                        gr.Button(new_suggestions[3] if len(new_suggestions) > 3 else "to"),
                        gr.Button(new_suggestions[4] if len(new_suggestions) > 4 else "and"),
                    ]
                
                def add_word_4(text):
                    suggestions = self.get_next_word_options(text, 5)
                    word = suggestions[3] if len(suggestions) > 3 else "to"
                    new_text = f"{text} {word}".strip()
                    new_suggestions = self.get_next_word_options(new_text, 5)
                    return [
                        new_text,
                        gr.Button(new_suggestions[0] if len(new_suggestions) > 0 else "the"),
                        gr.Button(new_suggestions[1] if len(new_suggestions) > 1 else "a"),
                        gr.Button(new_suggestions[2] if len(new_suggestions) > 2 else "is"),
                        gr.Button(new_suggestions[3] if len(new_suggestions) > 3 else "to"),
                        gr.Button(new_suggestions[4] if len(new_suggestions) > 4 else "and"),
                    ]
                
                def add_word_5(text):
                    suggestions = self.get_next_word_options(text, 5)
                    word = suggestions[4] if len(suggestions) > 4 else "and"
                    new_text = f"{text} {word}".strip()
                    new_suggestions = self.get_next_word_options(new_text, 5)
                    return [
                        new_text,
                        gr.Button(new_suggestions[0] if len(new_suggestions) > 0 else "the"),
                        gr.Button(new_suggestions[1] if len(new_suggestions) > 1 else "a"),
                        gr.Button(new_suggestions[2] if len(new_suggestions) > 2 else "is"),
                        gr.Button(new_suggestions[3] if len(new_suggestions) > 3 else "to"),
                        gr.Button(new_suggestions[4] if len(new_suggestions) > 4 else "and"),
                    ]
                
                # Wire up buttons
                outputs = [guided_text, word_btn_1, word_btn_2, word_btn_3, word_btn_4, word_btn_5]
                
                word_btn_1.click(fn=add_word_1, inputs=guided_text, outputs=outputs)
                word_btn_2.click(fn=add_word_2, inputs=guided_text, outputs=outputs)
                word_btn_3.click(fn=add_word_3, inputs=guided_text, outputs=outputs)
                word_btn_4.click(fn=add_word_4, inputs=guided_text, outputs=outputs)
                word_btn_5.click(fn=add_word_5, inputs=guided_text, outputs=outputs)
                
                refresh_btn.click(
                    fn=update_suggestions,
                    inputs=guided_text,
                    outputs=[word_btn_1, word_btn_2, word_btn_3, word_btn_4, word_btn_5]
                )
                
                clear_btn.click(
                    fn=lambda: ["", 
                                gr.Button("The"), gr.Button("I"), gr.Button("In"), 
                                gr.Button("A"), gr.Button("It")],
                    outputs=outputs
                )
                
                # Initialize buttons on load
                demo.load(
                    fn=lambda: [gr.Button("The"), gr.Button("I"), gr.Button("In"), 
                               gr.Button("A"), gr.Button("It")],
                    outputs=[word_btn_1, word_btn_2, word_btn_3, word_btn_4, word_btn_5]
                )
            
            # Footer
            gr.Markdown("""
            ---
            <div style="text-align: center; color: #666; font-size: 12px;">
                <p><strong>Smart Text Assistant</strong> | Powered by AI</p>
                <p>El Guelta Mohamad Saber · El Hadifi Soukaina</p>
            </div>
            """)
        
        return demo
    
    def launch(self, **kwargs):
        """Launch the application."""
        return self.demo.launch(**kwargs)


def main():
    """Main entry point."""
    print("=" * 60)
    print("✨ SMART TEXT ASSISTANT")
    print("   Professional AI-Powered Writing Tool")
    print("=" * 60)
    print()
    
    # Check if files exist
    corpus_path = "big_data.txt"
    keyboard_path = "qwerty_graph.txt"
    
    if not os.path.exists(corpus_path):
        print(f"❌ Error: Corpus file not found: {corpus_path}")
        return
    
    if not os.path.exists(keyboard_path):
        print(f"❌ Error: Keyboard graph file not found: {keyboard_path}")
        return
    
    try:
        # Initialize processor
        print("🚀 Initializing AI models...")
        print("   This may take a few minutes on first run...")
        print()
        
        processor = HybridTextProcessor(
            corpus_path=corpus_path,
            keyboard_graph_path=keyboard_path,
            ngram_size=2,
            smoothing_k=0.1,
            min_frequency=2,
            use_transformers=True
        )
        
        print()
        print("✅ AI models ready!")
        print()
        
        # Launch app
        print("🌐 Launching Smart Text Assistant...")
        app = SmartTextAssistant(processor)
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
