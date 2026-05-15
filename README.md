# AI-Driven-Compliance-Auditor
It created a Knowledge base with documents like CBUAE, CDD regulations and matches the uploaded document with the knowledge base and then uses LLM to print missing things and suggestions.
It also allows use to Query their uploaded document.
I also deployed this project on AWS ECS using docker and created the Knowledge BAse in Qdrant but ECS was too costly for a hobby project like this.
AWS Lambda could not have been used because embedding the uploaded PDF takes time.
