"""
Side-by-side demo for browser automation comparing standard vs adaptable agents.

This script can be run in two separate terminals:
- Terminal 1: python demo_browser_side_by_side.py --agent-type standard
- Terminal 2: python demo_browser_side_by_side.py --agent-type adaptable

The final LLM output will be printed clearly at the end for easy comparison.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from playwright.async_api import (
    async_playwright,
    Browser,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

# Add adaptable-agents-python-package to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "adaptable-agents-python-package")
)

from adaptable_agents import AdaptableOpenAIClient, ContextConfig
from openai import OpenAI

load_dotenv()


class BrowserAgent:
    """
    Browser automation agent using planner-executor architecture.

    The agent follows this loop:
    1. Planner (using AdaptableOpenAIClient) creates a high-level plan
    2. Executor (using standard OpenAI) executes plan steps one by one
    3. If execution fails or needs adjustment, re-plan using planner

    The planner uses AdaptableOpenAIClient to leverage past experiences,
    while the executor uses standard OpenAI for efficient step-by-step execution.
    """

    def __init__(
        self,
        planner_client: AdaptableOpenAIClient | OpenAI,
        executor_client: OpenAI,
        model: str = "gpt-5.1",
        max_steps: int = 20,
    ):
        self.planner_client = planner_client
        self.executor_client = executor_client
        self.model = model
        self.max_steps = max_steps
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        self.current_plan: list[dict[str, Any]] = []
        self.plan_index = 0
        self.final_llm_output = ""  # Store final LLM response

    async def observe_page(self, page: Page) -> str:
        """Extract observable information from the current page."""
        try:
            # Get page title and URL
            title = await page.title()
            url = page.url

            # Get visible text content (simplified - get main content)
            try:
                # Try to get main content area
                main_content = await page.locator(
                    "main, article, [role='main']"
                ).first.inner_text(timeout=2000)
            except Exception:
                # Fallback to body text
                main_content = await page.locator("body").inner_text()

            # Limit content length to avoid token limits
            if len(main_content) > 3000:
                main_content = main_content[:3000] + "... (truncated)"

            # Get visible links and buttons
            links = await page.locator("a[href], button").all()
            visible_elements = []
            for elem in links[:20]:  # Limit to first 20 elements
                try:
                    text = await elem.inner_text(timeout=500)
                    if text and text.strip():
                        elem_type = await elem.evaluate(
                            "el => el.tagName.toLowerCase()"
                        )
                        visible_elements.append(f"{elem_type}: {text.strip()}")
                except Exception:
                    pass

            # Get elements with event listeners
            elements_with_listeners = await page.evaluate("""
                () => {
                    const elements = [];
                    const allElements = document.querySelectorAll('*');
                    const seenSelectors = new Set();
                    
                    for (const el of allElements) {
                        // Skip if not visible
                        const rect = el.getBoundingClientRect();
                        const style = window.getComputedStyle(el);
                        if (rect.width === 0 && rect.height === 0) continue;
                        if (style.display === 'none' || style.visibility === 'hidden') continue;
                        
                        // Check for event listeners
                        const hasListeners = 
                            // Event handler attributes (most reliable)
                            el.hasAttribute('onclick') ||
                            el.hasAttribute('onchange') ||
                            el.hasAttribute('onmouseover') ||
                            el.hasAttribute('onmouseout') ||
                            el.hasAttribute('onmousedown') ||
                            el.hasAttribute('onmouseup') ||
                            el.hasAttribute('onkeydown') ||
                            el.hasAttribute('onkeyup') ||
                            el.hasAttribute('onfocus') ||
                            el.hasAttribute('onblur') ||
                            el.hasAttribute('onsubmit') ||
                            // Inline event handlers (check if property exists and is a function)
                            (typeof el.onclick === 'function') ||
                            (typeof el.onchange === 'function') ||
                            (typeof el.onmouseover === 'function') ||
                            (typeof el.onmouseout === 'function') ||
                            (typeof el.onmousedown === 'function') ||
                            (typeof el.onmouseup === 'function') ||
                            (typeof el.onkeydown === 'function') ||
                            (typeof el.onkeyup === 'function') ||
                            (typeof el.onfocus === 'function') ||
                            (typeof el.onblur === 'function') ||
                            (typeof el.onsubmit === 'function') ||
                            // Interactive roles
                            el.getAttribute('role') === 'button' ||
                            el.getAttribute('role') === 'link' ||
                            el.getAttribute('role') === 'tab' ||
                            el.getAttribute('role') === 'menuitem' ||
                            // Cursor pointer style (often indicates clickability)
                            (style.cursor === 'pointer' && (el.tagName === 'DIV' || el.tagName === 'SPAN' || el.tagName === 'P')) ||
                            // Common interactive data attributes
                            el.hasAttribute('data-action') ||
                            el.hasAttribute('data-click') ||
                            el.hasAttribute('data-toggle') ||
                            el.hasAttribute('data-target');
                        
                        if (hasListeners) {
                            // Generate a unique selector
                            let selector = '';
                            if (el.id) {
                                selector = `#${el.id}`;
                            } else if (el.className && typeof el.className === 'string') {
                                const classes = el.className.trim().split(/\\s+/).filter(c => c);
                                if (classes.length > 0) {
                                    selector = `.${classes[0]}`;
                                }
                            }
                            
                            if (!selector) {
                                selector = el.tagName.toLowerCase();
                            }
                            
                            // Make selector more unique by adding nth-child if needed
                            if (seenSelectors.has(selector)) {
                                const parent = el.parentElement;
                                if (parent) {
                                    const siblings = Array.from(parent.children);
                                    const index = siblings.indexOf(el);
                                    selector = `${selector}:nth-child(${index + 1})`;
                                }
                            }
                            seenSelectors.add(selector);
                            
                            // Get text content
                            const text = el.innerText || el.textContent || '';
                            const trimmedText = text.trim().substring(0, 100);
                            
                            // Get event listener types
                            const listenerTypes = [];
                            if (typeof el.onclick === 'function' || el.hasAttribute('onclick')) listenerTypes.push('click');
                            if (typeof el.onchange === 'function' || el.hasAttribute('onchange')) listenerTypes.push('change');
                            if (typeof el.onmouseover === 'function' || el.hasAttribute('onmouseover')) listenerTypes.push('mouseover');
                            if (typeof el.onmouseout === 'function' || el.hasAttribute('onmouseout')) listenerTypes.push('mouseout');
                            if (typeof el.onmousedown === 'function' || el.hasAttribute('onmousedown')) listenerTypes.push('mousedown');
                            if (typeof el.onmouseup === 'function' || el.hasAttribute('onmouseup')) listenerTypes.push('mouseup');
                            if (typeof el.onkeydown === 'function' || el.hasAttribute('onkeydown')) listenerTypes.push('keydown');
                            if (typeof el.onkeyup === 'function' || el.hasAttribute('onkeyup')) listenerTypes.push('keyup');
                            if (typeof el.onfocus === 'function' || el.hasAttribute('onfocus')) listenerTypes.push('focus');
                            if (typeof el.onblur === 'function' || el.hasAttribute('onblur')) listenerTypes.push('blur');
                            if (typeof el.onsubmit === 'function' || el.hasAttribute('onsubmit')) listenerTypes.push('submit');
                            
                            const role = el.getAttribute('role');
                            if (role) listenerTypes.push(`role:${role}`);
                            
                            if (style.cursor === 'pointer') listenerTypes.push('cursor:pointer');
                            
                            elements.push({
                                selector: selector,
                                tag: el.tagName.toLowerCase(),
                                text: trimmedText,
                                listeners: listenerTypes.length > 0 ? listenerTypes.join(', ') : 'interactive',
                                hasText: trimmedText.length > 0
                            });
                        }
                    }
                    
                    // Limit to first 30 elements to avoid token limits
                    return elements.slice(0, 30);
                }
            """)

            # Format elements with event listeners
            listener_elements = []
            for elem in elements_with_listeners:
                if elem.get("hasText") and elem.get("text"):
                    listener_elements.append(
                        f'{elem["tag"]} [{elem["selector"]}]: "{elem["text"][:50]}" (listeners: {elem["listeners"]})'
                    )
                else:
                    listener_elements.append(
                        f"{elem['tag']} [{elem['selector']}] (listeners: {elem['listeners']})"
                    )

            observation = f"""Page Title: {title}
                            URL: {url}

                            Main Content:
                            {main_content}

                            Visible Interactive Elements (links/buttons):
                            {chr(10).join(visible_elements[:10])}

                            Elements with Event Listeners:
                            {chr(10).join(listener_elements[:20])}"""

            return observation
        except Exception as e:
            return f"Error observing page: {str(e)}"

    async def execute_action(
        self, page: Page, action: dict[str, Any]
    ) -> tuple[str, bool]:
        """
        Execute a browser action.

        Returns:
            (result_message, task_complete)
        """
        action_type = action.get("action", "").lower()
        task_complete = action.get("task_complete", False)

        try:
            if action_type == "navigate":
                url = action.get("url", "")
                await page.goto(url, wait_until="networkidle", timeout=30000)
                return f"Navigated to {url}", task_complete

            elif action_type == "click":
                selector = action.get("selector", "")
                text = action.get("text", "")

                if selector:
                    await page.locator(selector).first.click(timeout=5000)
                    return f"Clicked element with selector: {selector}", task_complete
                elif text:
                    # Try to find element by text
                    await page.get_by_text(text, exact=False).first.click(timeout=5000)
                    return f"Clicked element with text: {text}", task_complete
                else:
                    return (
                        "Error: No selector or text provided for click action",
                        task_complete,
                    )

            elif action_type == "type":
                selector = action.get("selector", "")
                text = action.get("text", "")

                if selector:
                    await page.locator(selector).first.fill(text, timeout=5000)
                    return f"Typed '{text}' into {selector}", task_complete
                else:
                    return "Error: No selector provided for type action", task_complete

            elif action_type == "wait":
                seconds = action.get("seconds", 2)
                await page.wait_for_timeout(seconds * 1000)
                return f"Waited {seconds} seconds", task_complete

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                if direction == "down":
                    await page.evaluate("window.scrollBy(0, 500)")
                    return "Scrolled down", task_complete
                else:
                    await page.evaluate("window.scrollBy(0, -500)")
                    return "Scrolled up", task_complete

            elif action_type == "extract":
                # Task is complete, extract the answer
                selector = action.get("selector", "")
                if selector:
                    try:
                        # Wait for element to be visible first
                        await page.locator(selector).first.wait_for(
                            state="visible", timeout=10000
                        )
                        text = await page.locator(selector).first.inner_text(
                            timeout=5000
                        )
                        return f"Extracted text: {text}", True
                    except (PlaywrightTimeoutError, Exception) as e:
                        # Fall back to observation if selector extraction fails
                        observation = await self.observe_page(page)
                        return (
                            f"Extracted from page (selector failed: {str(e)}): {observation[:500]}",
                            True,
                        )
                else:
                    # Try to extract from observation
                    observation = await self.observe_page(page)
                    return f"Extracted from page: {observation[:500]}", True

            else:
                return f"Unknown action type: {action_type}", task_complete

        except PlaywrightTimeoutError:
            return f"Timeout executing action: {action_type}", task_complete
        except Exception as e:
            return f"Error executing action {action_type}: {str(e)}", task_complete

    async def create_plan(
        self, task: str, observation: str, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Use AdaptableOpenAIClient (planner) to create a plan.

        Returns a list of actions to execute.
        """
        system_prompt = """You are a browser automation planner. Your job is to create a detailed plan to complete tasks.

                            You can plan these actions:
                            1. navigate - Navigate to a URL: {"action": "navigate", "url": "https://example.com"}
                            2. click - Click an element: {"action": "click", "selector": "css-selector"} or {"action": "click", "text": "button text"}
                            3. type - Type text into an input: {"action": "type", "selector": "css-selector", "text": "text to type"}
                            4. wait - Wait for a few seconds: {"action": "wait", "seconds": 2}
                            5. scroll - Scroll the page: {"action": "scroll", "direction": "down"}
                            6. extract - Extract information (use when task is complete): {"action": "extract", "selector": "css-selector", "task_complete": true}

                            IMPORTANT: You must respond with a JSON object containing a "plan" array of actions.
                            The last action should have "task_complete": true if it completes the task.
                            Example: {"plan": [{"action": "navigate", "url": "https://example.com"}, {"action": "click", "text": "button"}, {"action": "extract", "task_complete": true}]}"""

        user_message = f"""Task: {task}

                        Current Page State:
                        {observation}

                        Previous Actions (if any):
                        {json.dumps(history[-3:], indent=2) if history else "None"}

                        Create a plan to complete this task. Respond with ONLY a JSON object containing a "plan" array."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Use planner client (AdaptableOpenAIClient)
        response = self.planner_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or '{"plan": []}'
        self.final_llm_output = content  # Store the final LLM output

        try:
            result = json.loads(content)
            plan = result.get("plan", [])
            if not isinstance(plan, list):
                plan = [plan]
            return plan
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    plan = result.get("plan", [])
                    if not isinstance(plan, list):
                        plan = [plan]
                    return plan
                except Exception:
                    pass
            # Fallback: return a simple wait action
            return [{"action": "wait", "seconds": 1, "task_complete": False}]

    async def decide_action(
        self,
        task: str,
        planner_step: dict[str, Any],
        observation: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | str:
        """
        Decide next action using executor (standard OpenAI).
        Takes a planner step and current page observation, then figures out the concrete action to execute.
        If extraction is needed, returns the extracted values directly as a string.

        Args:
            task: The overall task description
            planner_step: The high-level step from the planner
            observation: The observable page state from observe_page
            history: Previous action history

        Returns:
            Either a concrete action dict to execute, or a string with extracted values if extraction is needed
        """
        system_prompt = """You are a browser automation executor. Your job is to decide the next concrete action to execute based on a high-level planner step.

                        You can perform these actions:
                        1. navigate - Navigate to a URL: {"action": "navigate", "url": "https://example.com"}
                        2. click - Click an element: {"action": "click", "selector": "css-selector"} or {"action": "click", "text": "button text"}
                        3. type - Type text into an input: {"action": "type", "selector": "css-selector", "text": "text to type"}
                        4. wait - Wait for a few seconds: {"action": "wait", "seconds": 2}
                        5. scroll - Scroll the page: {"action": "scroll", "direction": "down"}

                        IMPORTANT: 
                        - You receive a high-level planner step that needs to be broken down into a concrete action
                        - Use the current page observation to find the exact selectors, text, or elements needed
                        - If the planner step requires extracting information or values from the page, you should directly extract and return those values as a plain string (not a JSON object)
                        - When extracting values, return ONLY the extracted information as a string, no JSON formatting needed
                        - For all other actions, respond with ONLY a valid JSON object containing the concrete action. No other text."""

        user_message = f"""Overall Task: {task}

                    Planner Step (High-level goal):
                    {json.dumps(planner_step, indent=2)}

                    Current Page State:
                    {observation}

                    Previous Actions:
                    {json.dumps(history[-3:], indent=2) if history else "None"}

                    Based on the planner step above and the current page state:
                    - If the step requires extracting information/values, extract them directly from the page observation and return ONLY the extracted values as a plain string
                    - Otherwise, decide the concrete action to execute and return a JSON object with the action
                    Use the page observation to find the exact selectors, text, or elements needed."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Use executor client (standard OpenAI)
        # Don't force JSON format - allow string responses for extraction
        response = self.executor_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

        content = response.choices[0].message.content or "{}"
        self.final_llm_output = content  # Store the final LLM output

        # Check if this is an extraction request (planner step indicates extraction)
        is_extraction = (
            planner_step.get("action") == "extract"
            or planner_step.get("task_complete", False)
            or "extract" in str(planner_step).lower()
        )

        # If it's an extraction, try to return as string first
        if is_extraction:
            # Check if content is already a plain string (not JSON)
            content_stripped = content.strip()
            if not content_stripped.startswith("{") and not content_stripped.startswith(
                "["
            ):
                # It's already a string, return it directly
                return content_stripped

        # Try to parse as JSON (for actions)
        try:
            action = json.loads(content)
            # Preserve task_complete from planner step if executor didn't set it
            if "task_complete" not in action and planner_step.get("task_complete"):
                action["task_complete"] = planner_step.get("task_complete", False)
            return action
        except json.JSONDecodeError:
            # If JSON parsing fails, it might be a string extraction
            if is_extraction:
                # Return the content as a string
                return content.strip()

            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{[^{}]*\}", content)
            if json_match:
                try:
                    action = json.loads(json_match.group())
                    if "task_complete" not in action and planner_step.get(
                        "task_complete"
                    ):
                        action["task_complete"] = planner_step.get(
                            "task_complete", False
                        )
                    return action
                except Exception:
                    pass
            # Fallback: use planner step if available, otherwise wait
            if planner_step:
                return planner_step
            return {"action": "wait", "seconds": 1, "task_complete": False}

    async def run(self, task: str, browser: Browser) -> dict[str, Any]:
        """
        Run the agent to complete the task using planner-executor architecture.

        Returns:
            Dictionary with success, result, steps, and history
        """
        page = await browser.new_page()
        self.step_count = 0
        self.history = []
        self.current_plan = []
        self.plan_index = 0
        self.final_llm_output = ""  # Reset final output
        task_complete = False
        result = ""
        consecutive_failures = 0
        max_consecutive_failures = 3

        try:
            while self.step_count < self.max_steps and not task_complete:
                # If we don't have a plan or finished current plan, create a new one
                if not self.current_plan or self.plan_index >= len(self.current_plan):
                    print("\n[Planner] Creating plan...")
                    observation = await self.observe_page(page)
                    self.current_plan = await self.create_plan(
                        task, observation, self.history
                    )
                    self.plan_index = 0
                    print(f"[Planner] Created plan with {len(self.current_plan)} steps")
                    print(f"[Planner] Plan: {self.current_plan}")
                    if not self.current_plan:
                        print("[Planner] Empty plan, using executor to decide action")
                        # Create a dummy planner step for the executor
                        dummy_planner_step = {
                            "action": "decide_next",
                            "description": "No plan available, decide next action",
                        }
                        executor_response = await self.decide_action(
                            task, dummy_planner_step, observation, self.history
                        )
                        # If executor returned a string (extracted values), handle it immediately
                        if isinstance(executor_response, str):
                            result = executor_response
                            task_complete = True
                            break
                        # Otherwise, add the action to the plan
                        self.current_plan = [executor_response]

                # Execute next step from plan
                if self.plan_index < len(self.current_plan):
                    self.step_count += 1
                    print(
                        f"\nStep {self.step_count}/{self.max_steps} (Plan step {self.plan_index + 1}/{len(self.current_plan)})..."
                    )

                    # Observe current state
                    observation = await self.observe_page(page)

                    # Get planner step from plan
                    planner_step = self.current_plan[self.plan_index]
                    # Handle case where planner_step might be a string (shouldn't happen, but be safe)
                    if isinstance(planner_step, str):
                        result = planner_step
                        task_complete = True
                        break
                    print(
                        f"  [Planner] High-level step: {planner_step.get('action', 'unknown')}"
                    )

                    # Use executor to decide concrete action based on planner step and observation
                    print("  [Executor] Deciding concrete action...")
                    executor_response = await self.decide_action(
                        task, planner_step, observation, self.history
                    )

                    # Check if executor returned extracted values (string) or an action (dict)
                    if isinstance(executor_response, str):
                        # Executor directly returned extracted values
                        print("  [Executor] Extracted values directly from page")
                        action_result = executor_response
                        task_complete = True
                        action = None  # No action was executed
                    else:
                        # Executor returned an action to execute
                        action = executor_response
                        print(
                            f"  [Executor] Concrete action: {action.get('action', 'unknown')}"
                        )

                        # Execute the concrete action
                        action_result, task_complete = await self.execute_action(
                            page, action
                        )

                    # Record in history
                    step_record = {
                        "step": self.step_count,
                        "observation": observation[:500],  # Truncate for storage
                        "planner_step": planner_step,
                        "executor_action": action if action else "extracted_values",
                        "result": action_result,
                        "from_plan": True,
                    }
                    self.history.append(step_record)

                    print(f" Result: {action_result[:100]}")

                    # Check if execution failed
                    if "Error" in action_result or "Timeout" in action_result:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            print("\n[Planner] Too many failures, re-planning...")
                            self.current_plan = []  # Force re-plan
                            consecutive_failures = 0
                    else:
                        consecutive_failures = 0
                        self.plan_index += 1

                    if task_complete:
                        result = action_result
                        break

                    # Small delay between steps
                    await asyncio.sleep(0.5)
                else:
                    # Plan exhausted, create new plan
                    self.current_plan = []

            success = task_complete and result != ""

        except Exception as e:
            print(f"Error during execution: {str(e)}")
            success = False
            result = str(e)
        finally:
            await page.close()

        return {
            "success": success,
            "result": result,
            "steps": self.step_count,
            "history": self.history,
            "final_llm_output": self.final_llm_output,
        }


async def run_with_standard_openai(task: str, model: str = "gpt-5.1") -> dict[str, Any]:
    """
    Run browser automation task using standard OpenAI client (no planner-executor).

    Args:
        task: The task description
        model: OpenAI model to use

    Returns:
        Dictionary with metrics: success, duration, tokens, response
    """
    print("\n" + "=" * 60)
    print("Running with STANDARD OpenAI Client (No Adaptable Agents)")
    print("=" * 60)

    start_time = time.time()
    success = False
    response_text = ""
    tokens_used = 0
    final_llm_output = ""

    try:
        # Initialize standard OpenAI client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # For standard mode, use same client for both planner and executor
        # (but we'll use planner-executor architecture with standard client)
        agent = BrowserAgent(
            planner_client=openai_client,  # Use standard client as planner too
            executor_client=openai_client,
            model=model,
        )

        # Launch browser
        print("Launching browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            try:
                # Run agent
                result = await agent.run(task, browser)
                success = result["success"]
                response_text = result["result"]
                final_llm_output = result.get("final_llm_output", "")
                tokens_used = result["steps"] * 100  # Rough estimate
            finally:
                await browser.close()

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        import traceback

        traceback.print_exc()
        response_text = error_msg

    duration = time.time() - start_time

    return {
        "success": success,
        "duration": duration,
        "tokens_estimated": int(tokens_used),
        "response": response_text[:200] + "..."
        if len(response_text) > 200
        else response_text,
        "steps": agent.step_count if "agent" in locals() else 0,
        "final_llm_output": final_llm_output,
    }


async def run_with_adaptable_agent(
    task: str,
    model: str = "gpt-5.1",
    memory_scope_path: str = "browser-playwright/planner-3",
    similarity_threshold: float = 0.5,
    max_items: int = 20,
) -> dict[str, Any]:
    """
    Run browser automation task using AdaptableOpenAIClient.

    Args:
        task: The task description
        model: OpenAI model to use
        memory_scope_path: Memory scope for adaptable agents
        similarity_threshold: Similarity threshold for context retrieval
        max_items: Maximum number of past experiences to use

    Returns:
        Dictionary with metrics: success, duration, tokens, response
    """
    print("\n" + "=" * 60)
    print("Running with ADAPTABLE OpenAI Client (With Adaptable Agents)")
    print("=" * 60)

    start_time = time.time()
    success = False
    response_text = ""
    tokens_used = 0
    final_llm_output = ""

    try:
        # Initialize AdaptableOpenAIClient
        adaptable_api_key = os.getenv("ADAPTABLE_API_KEY")
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

        if not adaptable_api_key:
            raise ValueError("ADAPTABLE_API_KEY not found in environment")

        context_config = ContextConfig(
            similarity_threshold=similarity_threshold,
            max_items=max_items,
        )

        adaptable_client = AdaptableOpenAIClient(
            adaptable_api_key=adaptable_api_key,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            api_base_url=api_base_url,
            memory_scope_path=memory_scope_path,
            context_config=context_config,
            auto_store_memories=True,
            enable_adaptable_agents=True,
            summarize_input=True,
        )

        # Initialize standard OpenAI client for executor
        executor_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize browser agent with planner-executor architecture
        # Planner uses AdaptableOpenAIClient, executor uses standard OpenAI
        agent = BrowserAgent(
            planner_client=adaptable_client,  # Adaptable agent for planning
            executor_client=executor_client,  # Standard client for execution
            model=model,
        )

        # Launch browser
        print("Launching browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            try:
                # Run agent
                result = await agent.run(task, browser)
                success = result["success"]
                response_text = result["result"]
                final_llm_output = result.get("final_llm_output", "")
                tokens_used = result["steps"] * 100  # Rough estimate
            finally:
                await browser.close()

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        import traceback

        traceback.print_exc()
        response_text = error_msg

    duration = time.time() - start_time

    return {
        "success": success,
        "duration": duration,
        "tokens_estimated": int(tokens_used),
        "response": response_text[:200] + "..."
        if len(response_text) > 200
        else response_text,
        "steps": agent.step_count if "agent" in locals() else 0,
        "final_llm_output": final_llm_output,
    }


# Single hard task for video demo
BROWSER_TASKS = [
    {
        "id": 15,
        "name": "GitHub Repository Insights",
        "description": "Navigate to a GitHub repository and extract insights information",
        "task": "Go to https://github.com/vercel/next.js, navigate to the Insights tab, then go to the Contributors section, and tell me the top 3 contributors by number of commits (usernames only)",
        "complexity": "Hard",
    },
]


async def main(agent_type: str = "standard"):
    """
    Run browser automation demo with specified agent type.

    Args:
        agent_type: Either "standard" or "adaptable"
    """
    print("\n" + "=" * 80)
    print(
        f"Browser Automation Demo: {'STANDARD' if agent_type == 'standard' else 'ADAPTABLE'} Agent"
    )
    print("=" * 80)

    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file or as an environment variable")
        return

    if agent_type == "adaptable":
        adaptable_api_key = os.getenv("ADAPTABLE_API_KEY")
        if not adaptable_api_key:
            print("ERROR: ADAPTABLE_API_KEY not found in environment")
            print("Please set it in your .env file or as an environment variable")
            print("Or use --agent-type standard to run with standard OpenAI client")
            return

    # Run the single task
    for task_info in BROWSER_TASKS:
        print("\n" + "=" * 80)
        print(f"TASK: {task_info['name']} ({task_info['complexity']})")
        print("=" * 80)
        print(f"Description: {task_info['description']}")
        print(f"Task: {task_info['task']}\n")

        # Run with specified agent type
        if agent_type == "standard":
            print(f"\nRunning with STANDARD OpenAI...")
            results = await run_with_standard_openai(task_info["task"])
        elif agent_type == "adaptable":
            print(f"\nRunning with ADAPTABLE Agent...")
            results = await run_with_adaptable_agent(
                task_info["task"],
                memory_scope_path="browser-playwright/planner-3",
            )
        else:
            print(f"ERROR: Unknown agent type: {agent_type}")
            print("Use --agent-type standard or --agent-type adaptable")
            return

        # Print results summary
        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Success: {'✓ YES' if results['success'] else '✗ NO'}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Steps: {results['steps']}")
        print(f"Result: {results['response']}")

        # Print final LLM output prominently
        print("\n" + "=" * 80)
        print("FINAL LLM OUTPUT")
        print("=" * 80)
        if results.get("final_llm_output"):
            print(results["final_llm_output"])
        else:
            print("(No LLM output captured)")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run browser automation demo with standard or adaptable agent"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["standard", "adaptable"],
        default="standard",
        help="Type of agent to run: 'standard' or 'adaptable' (default: standard)",
    )
    args = parser.parse_args()

    asyncio.run(main(agent_type=args.agent_type))
