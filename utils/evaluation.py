"""
Evaluation utilities for GameOf24 task.
"""

import re


def clean_output_for_GameOf24(output: str) -> str:
    """
    Clean the output for GameOf24 problems.
    """
    if "=" in output:
        output = output.split("=")[0].strip()
    if "is" in output:
        output = output.split("is")[1].strip()
    if "equals" in output:
        output = output.split("equals")[0].strip()
    if "evaluates to" in output:
        output = output.split("evaluates to")[0].strip()
    return output


def eval_for_GameOf24(input: str, output: str) -> bool:
    """
    Given an input and output, check if the output is correct and follows the rules of the game.
    
    Args:
        input: The input string with four numbers (e.g., "4 7 8 8")
        output: The model's output expression
        
    Returns:
        bool: True if the expression is correct and uses all numbers exactly once
    """
    clean_output = output

    clean_output = clean_output_for_GameOf24(output)
    clean_output = clean_output.replace("x", "*").strip()
    clean_output = clean_output.replace("×", "*").strip()
    clean_output = clean_output.replace("÷", "/").strip()
    
    try:
        # Get the value of the expression using eval
        value = eval(clean_output)
        if not (abs(value - 24) < 1e-3):
            return False
        
        # Split the input and output digits by space
        input_digits = input.split(" ")
        # Replace the following symbols with space
        replacements = ["+", "-", "*", "/", "÷", "(", ")"]
        for symbol in replacements:
            clean_output = clean_output.replace(symbol, " ")
        # Replace multiple spaces with single space
        clean_output = re.sub(" +", " ", clean_output)
        clean_output = clean_output.strip()
        output_digits = clean_output.split(" ")
        # Sort the digits
        input_digits.sort()
        output_digits.sort()
        # Check if the digits are the same
        if input_digits != output_digits:
            return False
        return True
    except Exception as e:
        return False
