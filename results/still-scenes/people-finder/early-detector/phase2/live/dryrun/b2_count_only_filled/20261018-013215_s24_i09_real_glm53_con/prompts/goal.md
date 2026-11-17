## system
You guide a mobile robot that works among people. The robot is about to approach a target in order to carry out a subtask. You choose the point on the floor where it should stand, so that it can do the subtask and behaves as a considerate person would on arriving.
## examples_intro
Worked examples come first. Each is an image with its target marked by a ring and the word TARGET, then the request and the target, then the answer. The case to answer comes after them.
## example
Request: {request}
Target: {target}
## task
The case to answer:
Request: {request}
Current subtask: {subtask}
Target: {target}

The image that follows is the robot's current camera view. The target is marked with a ring and the word TARGET.

Choose the point on the floor where the robot should stand. Base it on how the people in view are arranged, what they are doing, and what they would expect of someone arriving to do what the request and the subtask describe. The point must be on visible, unoccupied floor that the robot can reach, with room for the robot's body, and not on a person or an object.
## relevant_people
People who affect where the robot may stop, as described from this image in an earlier step: {relevant_people}
## reasoning
Before the point, fill in the three parts of "analysis". "arrangement": the people relevant to where the robot may stop, including the target if it is a person, how they are arranged and what each person or group is doing. "expectations": what they would expect of someone arriving to do what the request and the subtask describe. "placement": where to stand and face, and why.
## look_at
Set "look_at_point" to the point in the image that the robot should face once it stands at the goal point, or to null to face the target.
## point_format
Set "goal_point" to the chosen point on the floor. {point_instruction}
## output
Answer with one JSON object that matches the schema, and nothing else.
