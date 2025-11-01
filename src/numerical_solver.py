"""
Numerical Problem Solver using Chain-of-Thought prompting
Uses Groq API (fast and free tier available)
"""

import os
from typing import Dict
from dotenv import load_dotenv
from groq import Groq
from textbook_kb_builder import TextbookKnowledgeBase
import config

# Load environment variables from .env file
load_dotenv()


class NumericalSolver:
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize numerical solver with Groq API
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env variable)
            model: Model to use (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        self.model = model
        
        # Initialize knowledge base for context retrieval
        self.kb = TextbookKnowledgeBase()
        
        print(f"✅ Numerical Solver initialized with Groq model: {model}")
    
    def create_cot_prompt(self, question: str, context: str = "") -> str:
        """Create Chain-of-Thought prompt for numerical problem"""
        
        prompt = f"""You are an expert physics and mathematics tutor. Solve the following numerical problem step-by-step using chain-of-thought reasoning.

INSTRUCTIONS:
1. Break down the problem into clear, numbered steps
2. Show all equations and formulas used (without LaTeX formatting - use plain text)
3. Explain the reasoning for each step in simple language
4. Show calculations step-by-step with actual numbers
5. Provide the final answer clearly with proper units

IMPORTANT: Format your response using simple text formatting, NOT LaTeX. Use plain text equations like:
- Instead of \(v = u + at\), write: v = u + at
- Instead of \(a^2 + b^2 = c^2\), write: a^2 + b^2 = c^2
- Use multiple lines for clarity, not fancy formatting

"""
        
        if context:
            prompt += f"""RELEVANT TEXTBOOK CONTEXT:
{context}

"""
        
        prompt += f"""PROBLEM:
{question}

SOLUTION (step-by-step with clear numbering):"""
        
        return prompt
    
    def format_output(self, solution: str) -> str:
        """Format and clean up the solution for better readability"""
        
        # Remove LaTeX formatting
        solution = solution.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
        
        # Clean up multiple newlines
        lines = solution.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Remove extra whitespace but keep structure
            line = line.strip()
            if line:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def solve(self, question: str, use_context: bool = True) -> Dict:
        """
        Solve numerical problem using chain-of-thought reasoning
        
        Args:
            question: The numerical problem to solve
            use_context: Whether to retrieve relevant context from knowledge base
            
        Returns:
            Dictionary containing solution, reasoning steps, and metadata
        """
        print(f"\n{'='*70}")
        print("🔢 NUMERICAL PROBLEM SOLVER")
        print(f"{'='*70}\n")
        
        # Retrieve relevant context from knowledge base
        context = ""
        if use_context:
            print("📚 Retrieving relevant context from textbook...")
            context = self.kb.get_context_for_question(question, top_k=3)
            if context:
                print("✅ Context retrieved successfully\n")
            else:
                print("⚠️  No relevant context found\n")
        
        # Create chain-of-thought prompt
        prompt = self.create_cot_prompt(question, context)
        
        # Call Groq API
        print("🤖 Generating step-by-step solution...\n")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert physics and mathematics tutor who explains solutions in clear, understandable steps using plain text formatting (no LaTeX)."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            solution = response.choices[0].message.content
            
            # Format output for better readability
            solution = self.format_output(solution)
            
            print(f"{'='*70}")
            print("📝 SOLUTION:")
            print(f"{'='*70}\n")
            print(solution)
            print(f"\n{'='*70}\n")
            
            return {
                "question": question,
                "solution": solution,
                "context_used": context,
                "model": self.model,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error calling Groq API: {e}")
            return {
                "question": question,
                "solution": None,
                "error": str(e),
                "success": False
            }
    
    def interactive_solve(self):
        """Interactive mode for solving multiple problems"""
        print("\n🔢 Interactive Numerical Problem Solver (Groq)")
        print("Type 'exit' to quit\n")
        
        while True:
            question = input("Enter numerical problem: ").strip()
            if question.lower() == 'exit':
                break
            
            if not question:
                continue
            
            result = self.solve(question)
            
            if result['success']:
                feedback = input("\n💭 Was this solution helpful? (yes/no): ").strip().lower()
                if feedback in ['no', 'n']:
                    comments = input("What could be improved? ").strip()
                    print(f"📝 Feedback noted: {comments}")


def main():
    """Test the numerical solver"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Numerical Problem Solver with Groq API")
    parser.add_argument('--question', type=str, help='Numerical problem to solve')
    parser.add_argument('--api-key', type=str, help='Groq API key')
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile', help='Groq model to use')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        solver = NumericalSolver(api_key=args.api_key, model=args.model)
        
        if args.interactive:
            solver.interactive_solve()
        elif args.question:
            solver.solve(args.question)
        else:
            print("Please provide --question or use --interactive mode")
            
    except ValueError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
