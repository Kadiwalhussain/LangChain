"""
Runnable Advanced #2: Streaming
Real-time token-by-token output streaming
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import time
import sys

print("=" * 80)
print("STREAMING - REAL-TIME OUTPUT")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()

# Example 1: Basic streaming
print("\n1️⃣  Basic Streaming:")
print("-" * 80)

prompt = PromptTemplate.from_template("Tell me about {topic}:")
chain = prompt | llm | parser

print("\n🔄 Streaming output:\n")
print("-" * 80)

for chunk in chain.stream({"topic": "Python programming"}):
    print(chunk, end="", flush=True)
    time.sleep(0.01)  # Small delay to visualize streaming

print("\n" + "-" * 80)

# Example 2: Streaming with custom processor
print("\n\n2️⃣  Streaming with Custom Processing:")
print("-" * 80)

def uppercase_chunk(chunk: str) -> str:
    """Transform each chunk to uppercase"""
    return chunk.upper()

def add_markers(chunk: str) -> str:
    """Add markers to chunks"""
    return f"[{chunk}]"

# Chain with streaming transformations
streaming_chain = prompt | llm | parser | RunnableLambda(add_markers)

print("\n🔄 Streaming with markers:\n")
print("-" * 80)

chunk_count = 0
for chunk in streaming_chain.stream({"topic": "JavaScript"}):
    print(chunk, end="", flush=True)
    chunk_count += 1
    time.sleep(0.01)

print(f"\n\n📊 Total chunks received: {chunk_count}")
print("-" * 80)

# Example 3: Streaming with accumulation
print("\n\n3️⃣  Streaming with Accumulation:")
print("-" * 80)

print("\n🔄 Accumulating stream:\n")
print("-" * 80)

accumulated_text = ""
word_count = 0

for chunk in chain.stream({"topic": "Machine Learning"}):
    accumulated_text += chunk
    word_count = len(accumulated_text.split())
    
    # Print chunk with progress
    print(chunk, end="", flush=True)
    
    # Show word count every 10 words
    if word_count % 10 == 0 and chunk.strip():
        sys.stdout.write(f" [{word_count}]")
        sys.stdout.flush()
    
    time.sleep(0.01)

print(f"\n\n📊 Final statistics:")
print(f"   Total words: {word_count}")
print(f"   Total characters: {len(accumulated_text)}")
print("-" * 80)

# Example 4: Streaming multiple chains
print("\n\n4️⃣  Streaming Multiple Chains:")
print("-" * 80)

def stream_with_label(chain, topic: str, label: str):
    """Stream with label prefix"""
    print(f"\n{label}:")
    print("-" * 40)
    
    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
        time.sleep(0.01)
    
    print("\n")

topics_and_labels = [
    ("AI", "🤖 Artificial Intelligence"),
    ("Blockchain", "🔗 Blockchain")
]

for topic, label in topics_and_labels:
    stream_with_label(chain, topic, label)

# Example 5: Streaming with error handling
print("\n\n5️⃣  Streaming with Error Handling:")
print("-" * 80)

def safe_stream(chain, input_data: dict):
    """Stream with error handling"""
    try:
        print("\n🔄 Safe streaming:\n")
        print("-" * 80)
        
        chunk_count = 0
        error_chunks = 0
        
        for chunk in chain.stream(input_data):
            try:
                # Validate chunk
                if chunk and isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    chunk_count += 1
                else:
                    error_chunks += 1
                    
            except Exception as chunk_error:
                error_chunks += 1
                print(f"\n⚠️  Chunk error: {chunk_error}")
        
        print(f"\n\n✅ Streaming completed:")
        print(f"   Valid chunks: {chunk_count}")
        print(f"   Error chunks: {error_chunks}")
        
    except Exception as e:
        print(f"\n❌ Stream error: {e}")

safe_stream(chain, {"topic": "Cloud Computing"})

# Example 6: Batch streaming (multiple inputs)
print("\n\n6️⃣  Batch Streaming:")
print("-" * 80)

topics = ["Python", "JavaScript", "Go"]

print("\n🔄 Streaming batch processing:\n")

for i, topic in enumerate(topics, 1):
    print(f"\n{i}. Topic: {topic}")
    print("-" * 40)
    
    # Stream each topic
    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
        time.sleep(0.005)  # Faster streaming for batch
    
    print("\n")

# Example 7: Streaming with progress indicators
print("\n\n7️⃣  Streaming with Progress:")
print("-" * 80)

def stream_with_progress(chain, input_data: dict, target_words: int = 50):
    """Stream with progress bar"""
    print("\n🔄 Streaming with progress:\n")
    print("-" * 80)
    
    accumulated = ""
    
    for chunk in chain.stream(input_data):
        accumulated += chunk
        current_words = len(accumulated.split())
        
        # Print chunk
        print(chunk, end="", flush=True)
        
        # Calculate progress
        progress = min(100, int((current_words / target_words) * 100))
        
        # Update progress indicator every 10%
        if current_words % 5 == 0:
            bar_length = 20
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            sys.stdout.write(f"\n[{bar}] {progress}%\n")
            sys.stdout.flush()
        
        time.sleep(0.01)
    
    print("\n\n✅ Complete!")

stream_with_progress(chain, {"topic": "Data Science"}, target_words=30)

# Example 8: Streaming comparison (stream vs invoke)
print("\n\n8️⃣  Performance Comparison:")
print("-" * 80)

print("\n⏱️  Testing invoke (blocking):")
start_invoke = time.time()
result_invoke = chain.invoke({"topic": "AI"})
time_invoke = time.time() - start_invoke
print(f"   Time: {time_invoke:.2f}s")
print(f"   First output: {time_invoke:.2f}s (waits for complete response)")

print("\n⏱️  Testing stream (non-blocking):")
start_stream = time.time()
first_chunk_time = None
chunks = []

for i, chunk in enumerate(chain.stream({"topic": "AI"})):
    if i == 0 and first_chunk_time is None:
        first_chunk_time = time.time() - start_stream
    chunks.append(chunk)

time_stream = time.time() - start_stream

print(f"   Total time: {time_stream:.2f}s")
print(f"   First output: {first_chunk_time:.3f}s (immediate streaming)")
print(f"   Perceived speed improvement: {(time_invoke/first_chunk_time):.1f}x faster")

print("\n" + "=" * 80)
print("STREAMING EXPLAINED")
print("=" * 80)
print("""
Streaming provides real-time token-by-token output:

Methods:

1. .stream(input)
   - Synchronous streaming
   - Returns generator
   - Yields chunks as they arrive

2. .astream(input)
   - Async streaming
   - For async applications
   - Non-blocking

Syntax:
    for chunk in chain.stream(input):
        print(chunk, end="")

vs Regular:
    result = chain.invoke(input)
    print(result)

Benefits:
✅ Lower perceived latency
✅ Better UX (immediate feedback)
✅ Progressive rendering
✅ Early error detection
✅ Memory efficient
✅ Real-time processing

Use Cases:
- Chatbots (ChatGPT-style)
- Live transcription
- Long-form content
- Progressive summaries
- Real-time translation
- Interactive applications

Implementation Patterns:

A. Simple Stream:
    for chunk in chain.stream(input):
        display(chunk)

B. Accumulating Stream:
    buffer = ""
    for chunk in chain.stream(input):
        buffer += chunk
        process(buffer)

C. Transform Stream:
    for chunk in chain.stream(input):
        transformed = transform(chunk)
        display(transformed)

D. Multi-Stream:
    for source, chunk in multi_chain.stream(input):
        display(source, chunk)

Best Practices:
- Flush output buffers
- Handle partial tokens
- Graceful error handling
- Show progress indicators
- Validate chunks
- Test with slow connections

Performance:
- First token: ~100-500ms
- vs Full response: 2-10s
- User perception: 5-10x faster

Streaming + Other Features:
- ✅ Parallel chains (each streams)
- ✅ Fallbacks (stream from fallback)
- ✅ Retries (restart stream)
- ✅ Custom processing (transform chunks)
""")
print("=" * 80)
