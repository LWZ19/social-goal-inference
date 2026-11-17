## system
## task
Request: {request}
Current subtask: {subtask}
Target: {target}

The image that follows is the robot's current camera view. The target is marked with a ring and the word TARGET.

In "relevant_people", describe the people whose position, facing or activity affects where an arriving robot may stop or what it would block by stopping there. They can be at any distance from the target, and they include the target itself when it is a person. If there are none, write "none".

The standard goal is to stop about {standoff_m} m in front of the side of the target that the robot sees, squarely facing it. Set "stop_elsewhere" to false only when a considerate person arriving for the same purpose would go and stop there; otherwise set it to true. That person would not do so while others are already using the target, waiting for it, watching it or gathered with it, even if they stand to one side and leave that spot free. People who work at the target, or have nothing to do with it and would not be disturbed, make no difference.
## reasoning
Before the answer, fill in "reasoning": what the people in view are doing and whether it concerns the target, where the standard goal would put the robot, and what a considerate person arriving for the same purpose would do on seeing them.
## output
Answer with one JSON object that matches the schema, and nothing else.
