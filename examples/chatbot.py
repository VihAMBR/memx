"""
Interactive chatbot with persistent memory.

Run: python examples/chatbot.py --user alice
"""
import argparse
import os
from openai import OpenAI
from memx import MemorySystem


def make_openai_llm(model: str = "gpt-4o-mini"):
    client = OpenAI()

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""

    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="default", help="User ID for memory")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    llm = make_openai_llm(args.model)
    mem = MemorySystem(user_id=args.user, llm=llm)

    print(f"Chatbot ready. Memories: {mem.memory_count}. User: {args.user}")
    print("Type 'quit' to exit, 'context <query>' to see raw context.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                break

            # Debug command: show raw context
            if user_input.lower().startswith("context "):
                query = user_input[8:]
                print(f"\n--- Context for '{query}' ---")
                print(mem.get_context(query) or "(empty)")
                print("---\n")
                continue

            # Store user message
            mem.add("user", user_input)

            # Get memory context and build a full prompt
            context = mem.get_context(user_input)
            system_prompt = "You are a helpful assistant with memory of past conversations."
            if context:
                system_prompt += f"\n\nMemory:\n{context}"

            # Generate response
            response = llm(
                f"System: {system_prompt}\n\nUser: {user_input}\n\nAssistant:"
            )
            print(f"Bot: {response}\n")

            # Store assistant response
            mem.add("assistant", response)

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\nSaving session...")
        mem.end_session()
        mem.close()
        print(f"Done. {mem.memory_count} memories stored.")


if __name__ == "__main__":
    main()
