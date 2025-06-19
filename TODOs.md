# Random TODOs as I think of them

1. 03 poc quickstart i added if you run a few times, and I've seen this with other quickstarts then you get a 429 here and there, so we need to handle this more gracefully. Are we handling max iterations like i think we should be? Can we make this smarter and check based on the llm provider the limits to warn against so this doesn't surprise users?



2. Can I remove any extra logic since Agent class no longer inherits from AgentBase class? Or, should I add a new AgentBase to give back my validations?

3. Fix: if I put reasoning=true in the agent instantiation, then it goes to react agent, but the would some of the other agents use reasoning, should reasoning be moved to llm section instead of agent??


External Config Questions

- Quickstarts I've added 0-3 do not "need" dapr, but then I have to have llm config in a yaml file. Dapr gives us provider components. Do we want to support both? Or, make folks use dapr then??