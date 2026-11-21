---
tags: [syntra-dev]
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

add an endpoint POST /api/annotations/import that takes a path to a geojson file of labelled polygons and an image_id, and saves them as annotations for that image
