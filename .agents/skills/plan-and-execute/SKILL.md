---
name: plan-and-execute
description: A workflow that splits tasks into a Planning phase (using Claude/reasoning models) and an Execution phase (using Gemini/high-limit models) with safeguards against accidental execution.
---

# Plan and Execute Workflow

This skill facilitates a two-step workflow: using a reasoning model (like Claude Opus) for architectural planning, and seamlessly handing off the actual coding implementation to a high-limit model (like Gemini 3.1 Pro).

## Phase 1: Planning

When the user asks you to plan a feature or you activate this skill:
1. **Analyze and Think:** Carefully analyze the user's requirements, constraints, and the existing codebase.
2. **Draft the Plan:** Formulate a detailed, step-by-step implementation plan. Include specific files to edit, functions to create, and architectural decisions.
3. **Store the Plan:** Save the detailed plan into an artifact file named `implementation_plan.md` in the current conversation's artifact directory. This ensures the next model has a permanent, exact record of what to do.
4. **Handoff Prompt:** Stop working and provide the user with clear instructions to switch models. Use a message similar to this:

   > **Plan Ready!** I have saved the step-by-step implementation plan to `implementation_plan.md`. 
   > 
   > Please manually switch your model to **Gemini 3.1 Pro (High)** (or your preferred execution model) using the IDE settings.
   > 
   > Once you have switched, simply reply with: *"Please execute the plan in implementation_plan.md"*

## Execution Guardrail (Model Warning)

If the user asks the planning model to begin writing the implementation code *without* switching models, you **MUST NOT** proceed immediately.

1. **Pause and Warn:** Display a clear warning asking for confirmation:
   > [!WARNING]
   > **Are you sure you want to implement the plan using the current model?** 
   > You usually prefer to switch to a higher-limit model (like Gemini 3.1 Pro) for the actual coding phase. 
2. **Await Confirmation:** Only proceed with the implementation if the user explicitly confirms they want the current model to do the coding.

## Phase 2: Execution (For the Implementation Model)

When the user switches the model and asks to execute the plan:
1. **Read the Artifact:** Use your tools to read the contents of `implementation_plan.md`.
2. **Execute:** Follow the steps in the plan strictly and carefully. You do not need to re-think the architecture unless you hit a critical blocker.
3. **Verify:** Use appropriate commands or checks to ensure the implementation matches the success criteria defined in the plan.
