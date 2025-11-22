"""
Demo comparing browser automation performance with and without Adaptable Agents.

This demo runs the same browser automation task using:
1. Standard OpenAI client (without adaptable agents)
2. AdaptableOpenAIClient (with adaptable agents)

The task: "Find the number of stars of the browser-use repo on GitHub"

Uses Playwright for browser automation and OpenAI client directly.
"""

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
    Simple browser automation agent using Playwright and OpenAI.

    The agent follows a simple loop:
    1. Observe the current page state
    2. Use LLM to decide next action
    3. Execute the action
    4. Repeat until task is complete
    """

    def __init__(
        self,
        client: OpenAI | AdaptableOpenAIClient,
        model: str = "gpt-4o-mini",
        max_steps: int = 20,
    ):
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.step_count = 0
        self.history: list[dict[str, Any]] = []

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
                        f"{elem['tag']} [{elem['selector']}]: \"{elem['text'][:50]}\" (listeners: {elem['listeners']})"
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

    async def decide_action(
        self, task: str, observation: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Use LLM to decide the next action based on current state."""
        # Build messages
        system_prompt = """You are a browser automation assistant. Your job is to help complete tasks by controlling a web browser.

You can perform these actions:
1. navigate - Navigate to a URL: {"action": "navigate", "url": "https://example.com"}
2. click - Click an element: {"action": "click", "selector": "css-selector"} or {"action": "click", "text": "button text"}
3. type - Type text into an input: {"action": "type", "selector": "css-selector", "text": "text to type"}
4. wait - Wait for a few seconds: {"action": "wait", "seconds": 2}
5. scroll - Scroll the page: {"action": "scroll", "direction": "down"}
6. extract - Extract information (use when task is complete): {"action": "extract", "selector": "css-selector", "task_complete": true}

IMPORTANT: You must respond with ONLY a valid JSON object containing the action. No other text.
When the task is complete, set "task_complete": true in your response."""

        user_message = f"""Task: {task}

Current Page State:
{observation}

Previous Actions:
{json.dumps(history[-3:], indent=2) if history else "None"}

What action should I take next? Respond with ONLY a JSON object with the action."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"

        try:
            action = json.loads(content)
            return action
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{[^{}]*\}", content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    pass
            # Fallback action
            return {"action": "wait", "seconds": 1, "task_complete": False}

    async def run(self, task: str, browser: Browser) -> dict[str, Any]:
        """
        Run the agent to complete the task.

        Returns:
            Dictionary with success, result, steps, and history
        """
        page = await browser.new_page()
        self.step_count = 0
        self.history = []
        task_complete = False
        result = ""

        try:
            while self.step_count < self.max_steps and not task_complete:
                self.step_count += 1
                print(f"Step {self.step_count}/{self.max_steps}...")

                # Observe current state
                observation = await self.observe_page(page)

                # Decide next action
                action = await self.decide_action(task, observation, self.history)

                # Execute action
                action_result, task_complete = await self.execute_action(page, action)

                # Record in history
                step_record = {
                    "step": self.step_count,
                    "observation": observation[:500],  # Truncate for storage
                    "action": action,
                    "result": action_result,
                }
                self.history.append(step_record)

                print(f"  Action: {action.get('action', 'unknown')}")
                print(f"  Result: {action_result[:100]}")

                if task_complete:
                    result = action_result
                    break

                # Small delay between steps
                await asyncio.sleep(0.5)

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
        }


async def run_with_standard_openai(
    task: str, model: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """
    Run browser automation task using standard OpenAI client.

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

    try:
        # Initialize standard OpenAI client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize browser agent
        agent = BrowserAgent(openai_client, model=model)

        # Launch browser
        print("Launching browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            try:
                # Run agent
                result = await agent.run(task, browser)
                success = result["success"]
                response_text = result["result"]
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
    }


async def run_with_adaptable_agent(
    task: str,
    model: str = "gpt-4o-mini",
    memory_scope_path: str = "browser-playwright/github-tasks",
    similarity_threshold: float = 0.8,
    max_items: int = 5,
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
        )

        # Initialize browser agent
        agent = BrowserAgent(adaptable_client, model=model)

        # Launch browser
        print("Launching browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            try:
                # Run agent
                result = await agent.run(task, browser)
                success = result["success"]
                response_text = result["result"]
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
    }


# Define 5 multistep GitHub tasks with increasing complexity
BROWSER_TASKS = [
    {
        "id": 1,
        "name": "GitHub Repo Stars",
        "description": "Navigate to a GitHub repository and find the number of stars",
        "task": "Go to https://github.com/browser-use/browser-use and find the number of stars for this repository",
        "complexity": "Easy",
    },
    {
        "id": 2,
        "name": "GitHub Repo Stats",
        "description": "Navigate to a GitHub repository and extract multiple statistics",
        "task": "Go to https://github.com/browser-use/browser-use, find the number of stars, forks, and the primary programming language used in this repository",
        "complexity": "Easy-Medium",
    },
    {
        "id": 3,
        "name": "GitHub Issues Count",
        "description": "Navigate to a GitHub repository, go to issues tab, and count open issues",
        "task": "Go to https://github.com/browser-use/browser-use, navigate to the Issues tab, and tell me how many open issues there are",
        "complexity": "Medium",
    },
    {
        "id": 4,
        "name": "GitHub Latest Release",
        "description": "Navigate to a GitHub repository, go to releases, and find the latest release version",
        "task": "Go to https://github.com/browser-use/browser-use, navigate to the Releases section, and tell me what the latest release version is and when it was published",
        "complexity": "Medium-Hard",
    },
    {
        "id": 5,
        "name": "GitHub Contributor Profile",
        "description": "Navigate to a GitHub repository, find top contributor, and get their profile information",
        "task": "Go to https://github.com/browser-use/browser-use, navigate to the Contributors section, click on the top contributor's profile, and tell me their username, number of followers, and their location (if available)",
        "complexity": "Hard",
    },
]


async def main():
    """Run comparison demo with 5 multistep GitHub tasks of increasing complexity."""
    print("\n" + "=" * 80)
    print("Browser Automation Performance Comparison: Standard vs Adaptable Agent")
    print("5 Multistep GitHub Tasks with Increasing Complexity")
    print("=" * 80)

    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file or as an environment variable")
        return

    adaptable_api_key = os.getenv("ADAPTABLE_API_KEY")
    if not adaptable_api_key:
        print(
            "WARNING: ADAPTABLE_API_KEY not found. Running only standard OpenAI demo."
        )
        print(
            "Set ADAPTABLE_API_KEY in your .env file to enable adaptable agents comparison."
        )

    # Store results for all tasks
    all_results = []

    # Run each task
    for task_info in BROWSER_TASKS:
        print("\n" + "=" * 80)
        print(
            f"TASK {task_info['id']}/5: {task_info['name']} ({task_info['complexity']})"
        )
        print("=" * 80)
        print(f"Description: {task_info['description']}")
        print(f"Task: {task_info['task']}\n")

        task_result = {
            "task_id": task_info["id"],
            "task_name": task_info["name"],
            "complexity": task_info["complexity"],
            "description": task_info["description"],
            "task": task_info["task"],
        }

        # Run with standard OpenAI
        print(f"\n[Task {task_info['id']}] Running with STANDARD OpenAI...")
        standard_results = await run_with_standard_openai(task_info["task"])
        task_result["standard"] = standard_results

        # Run with adaptable agent (if API key is available)
        # Use shared memory scope path so all tasks can learn from each other
        if adaptable_api_key:
            print(f"\n[Task {task_info['id']}] Running with ADAPTABLE Agent...")
            adaptable_results = await run_with_adaptable_agent(
                task_info["task"],
                memory_scope_path="browser-playwright/github-tasks",
            )
            task_result["adaptable"] = adaptable_results
        else:
            task_result["adaptable"] = None

        all_results.append(task_result)

        # Small delay between tasks
        await asyncio.sleep(1)

    # Print comprehensive comparison
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 80)

    if adaptable_api_key:
        # Detailed comparison table
        print(
            f"\n{'Task':<25} {'Complexity':<15} {'Standard':<30} {'Adaptable':<30} {'Improvement':<15}"
        )
        print("-" * 115)
        print(
            f"{'ID':<5} {'Name':<20} {'':<15} {'Success':<10} {'Duration':<10} {'Steps':<10} {'Success':<10} {'Duration':<10} {'Steps':<10} {'Time %':<15}"
        )
        print("-" * 115)

        total_standard_success = 0
        total_adaptable_success = 0
        total_standard_duration = 0
        total_adaptable_duration = 0
        total_standard_steps = 0
        total_adaptable_steps = 0

        for result in all_results:
            std = result["standard"]
            ada = result.get("adaptable")

            if ada:
                std_success = "✓" if std["success"] else "✗"
                ada_success = "✓" if ada["success"] else "✗"
                std_duration = f"{std['duration']:.1f}s"
                ada_duration = f"{ada['duration']:.1f}s"
                std_steps = str(std.get("steps", 0))
                ada_steps = str(ada.get("steps", 0))

                if std["duration"] > 0:
                    improvement = (
                        (std["duration"] - ada["duration"]) / std["duration"]
                    ) * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"

                print(
                    f"{result['task_id']:<5} {result['task_name']:<20} {result['complexity']:<15} "
                    f"{std_success:<10} {std_duration:<10} {std_steps:<10} "
                    f"{ada_success:<10} {ada_duration:<10} {ada_steps:<10} {improvement_str:<15}"
                )

                if std["success"]:
                    total_standard_success += 1
                if ada["success"]:
                    total_adaptable_success += 1
                total_standard_duration += std["duration"]
                total_adaptable_duration += ada["duration"]
                total_standard_steps += std.get("steps", 0)
                total_adaptable_steps += ada.get("steps", 0)
            else:
                std_success = "✓" if std["success"] else "✗"
                std_duration = f"{std['duration']:.1f}s"
                std_steps = str(std.get("steps", 0))
                print(
                    f"{result['task_id']:<5} {result['task_name']:<20} {result['complexity']:<15} "
                    f"{std_success:<10} {std_duration:<10} {std_steps:<10} "
                    f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<15}"
                )
                if std["success"]:
                    total_standard_success += 1
                total_standard_duration += std["duration"]
                total_standard_steps += std.get("steps", 0)

        # Summary statistics
        num_tasks = len(all_results)
        print("\n" + "=" * 115)
        print("SUMMARY STATISTICS")
        print("=" * 115)
        print(
            f"\n{'Metric':<30} {'Standard Agent':<25} {'Adaptable Agent':<25} {'Improvement':<20}"
        )
        print("-" * 100)
        print(
            f"{'Success Rate':<30} {total_standard_success}/{num_tasks} ({total_standard_success/num_tasks*100:.1f}%){'':<10} "
            f"{total_adaptable_success}/{num_tasks} ({total_adaptable_success/num_tasks*100:.1f}%){'':<10} "
            f"{((total_adaptable_success - total_standard_success) / num_tasks * 100):+.1f}%"
        )
        avg_std_duration = total_standard_duration / num_tasks if num_tasks > 0 else 0
        avg_ada_duration = total_adaptable_duration / num_tasks if num_tasks > 0 else 0
        duration_improvement = (
            ((avg_std_duration - avg_ada_duration) / avg_std_duration * 100)
            if avg_std_duration > 0
            else 0
        )
        print(
            f"{'Average Duration':<30} {avg_std_duration:.2f}s{'':<15} {avg_ada_duration:.2f}s{'':<15} "
            f"{duration_improvement:+.1f}%"
        )
        avg_std_steps = total_standard_steps / num_tasks if num_tasks > 0 else 0
        avg_ada_steps = total_adaptable_steps / num_tasks if num_tasks > 0 else 0
        steps_improvement = (
            ((avg_std_steps - avg_ada_steps) / avg_std_steps * 100)
            if avg_std_steps > 0
            else 0
        )
        print(
            f"{'Average Steps':<30} {avg_std_steps:.1f}{'':<18} {avg_ada_steps:.1f}{'':<18} "
            f"{steps_improvement:+.1f}%"
        )
        total_duration_improvement = (
            (
                (total_standard_duration - total_adaptable_duration)
                / total_standard_duration
                * 100
            )
            if total_standard_duration > 0
            else 0
        )
        print(
            f"{'Total Duration':<30} {total_standard_duration:.2f}s{'':<15} {total_adaptable_duration:.2f}s{'':<15} "
            f"{total_duration_improvement:+.1f}%"
        )

        # Overall verdict
        print("\n" + "=" * 115)
        if total_adaptable_success > total_standard_success:
            print("✅ Adaptable Agent achieved HIGHER success rate!")
        elif total_adaptable_success == total_standard_success:
            print("➡️  Both agents achieved the same success rate")
        else:
            print(
                "⚠️  Standard Agent achieved higher success rate (adaptable agent may improve with more runs)"
            )

        if total_adaptable_duration < total_standard_duration:
            print("✅ Adaptable Agent completed tasks FASTER on average!")
        elif total_adaptable_duration > total_standard_duration:
            print(
                "⚠️  Standard Agent was faster (adaptable agent may improve with more runs)"
            )
        else:
            print("➡️  Similar average performance")

    else:
        # Only standard results available
        print(f"\n{'Task':<25} {'Complexity':<15} {'Standard Agent Results':<50}")
        print("-" * 90)
        print(
            f"{'ID':<5} {'Name':<20} {'':<15} {'Success':<10} {'Duration':<15} {'Steps':<10}"
        )
        print("-" * 90)

        total_standard_success = 0
        total_standard_duration = 0
        total_standard_steps = 0

        for result in all_results:
            std = result["standard"]
            std_success = "✓" if std["success"] else "✗"
            std_duration = f"{std['duration']:.1f}s"
            std_steps = str(std.get("steps", 0))
            print(
                f"{result['task_id']:<5} {result['task_name']:<20} {result['complexity']:<15} "
                f"{std_success:<10} {std_duration:<15} {std_steps:<10}"
            )
            if std["success"]:
                total_standard_success += 1
            total_standard_duration += std["duration"]
            total_standard_steps += std.get("steps", 0)

        num_tasks = len(all_results)
        print("\n" + "=" * 90)
        print("SUMMARY STATISTICS (Standard Agent Only)")
        print("=" * 90)
        print(
            f"Success Rate: {total_standard_success}/{num_tasks} ({total_standard_success/num_tasks*100:.1f}%)"
        )
        print(f"Average Duration: {total_standard_duration/num_tasks:.2f}s")
        print(f"Average Steps: {total_standard_steps/num_tasks:.1f}")
        print(f"Total Duration: {total_standard_duration:.2f}s")

    print("\n" + "=" * 80)
    print(
        "Note: Adaptable Agents learn from each run, so performance improves over time!"
    )
    print("Run this demo multiple times to see the improvement.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
