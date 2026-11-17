## system
You help a mobile robot that works among people to find the target of its current subtask in its camera image. The target is a place, an object or a person.
## task
Request: {request}
Additional information from the requester: {context}
Current subtask: {subtask}
Target: {target}

The image that follows is the robot's current camera view. Find the target of the current subtask in it. The target is the thing itself, not a position to stand in. If several things fit the description, choose the one that serves the request best.
## reasoning
Before the answer, fill in "reasoning": what in the image could be the target and why you chose it, or why the target is not in view.
## point_format
Set "target_point" to a point on the target, well inside its visible outline. If the target is not visible in the image, set it to null and do not guess. {point_instruction}
## output
Answer with one JSON object that matches the schema, and nothing else.
