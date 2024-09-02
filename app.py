import time
import gradio as gr
from xsbpe.basic import BasicTokenizer

tk = BasicTokenizer()
print('Tokenizer initialized.')
st = time.time()
tk.load('dune-20256.model')
et = time.time()
print(f'Model loaded. Took {et-st} seconds.')

def tokenize(text, checked):
    tokens = tk.encode(text)
    
    colors = ['rgba(107,64,216,.3)', 'rgba(104,222,122,.4)', 'rgba(244,172,54,.4)', 'rgba(239,65,70,.4)', 'rgba(39,181,234,.4)']
    colored_tokens = []
    
    for i, token in enumerate(tokens):
        token = tk.vocab[token].decode('utf-8').replace(' ', '⋅' if checked else '&nbsp;')
        span = f'<span style="background-color: {colors[i % len(colors)]}">{token}</span>'
        colored_tokens.append(span)

    return '<p style="margin-left: 2px; margin-right: 2px; word-wrap: break-word">' + ''.join(colored_tokens) + '</p>', tokens, len(tokens), len(text)

interface = gr.Interface(
    fn=tokenize, 
    inputs=[
        gr.TextArea(label='Input Text', type='text'),
        gr.Checkbox(label='Show whitespace')
    ], 
    outputs=[
        gr.HTML(label='Tokenized Text'),
        gr.Textbox(label='Token IDs', lines=1, max_lines=5),
        gr.Textbox(label='Tokens', max_lines=1),
        gr.Textbox(label='Characters', max_lines=1)
    ],
    title="BPE Tokenization Visualizer",
    live=True,
    examples=[
        ['BPE, or Byte Pair Encoding, is a method used to compress text by breaking it down into smaller units. In natural language processing, it helps tokenize words by merging the most frequent pairs of characters or symbols, creating more efficient and manageable tokens for analysis.', False],
        ['This custom BPE tokenizer model was trained on the entire text of the novel Dune by Frank Herbert and has a vocabulary size of 20,256, which corresponds to the 256 bytes base tokens and the symbols learned with 20,000 merges.', False],
        ['The spice must flow, Paul. Without it, the Fremen will never rise, and the sands will consume us all.', False]
    ],
    show_progress='hidden',
    api_name='tokenize',
    allow_flagging='never'
).launch(inbrowser=True)