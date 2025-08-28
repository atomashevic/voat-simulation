# Voat simulations

30-day sample targets:

- users ~ 715
- comments / post ~ 1.1
- interactions/user ~ 1.3
- mean toxicity ~ 0.1

## Latest config

```json
"starting_agents": 50,
"percentage_new_agents_iteration": 0.55,
"percentage_removed_agents_interation": 0.90,
"actions_likelihood": {
  "post": 0.005,
  "comment": 0.10,
  "read": 0.4,
  "search": 0.1,
  "share_link": 0.06
}
```

This config is based on following details from the Voat MADOC samples:

1. Expected user count ~ 720
2. High absolute churn ~ 80-85%
3. Moderate new user count 53-55%
4. Sample starts with ~50-55 users
5. Almost even number of posts and comemnts 793/704 864/819 808/735
6. 1.3 interactions per user on average
7. Toxicity is the same as in Reddit


This percentage of new agents is huuuuge!!! It leads to about 16k users which is not feasible we need love user count.

```json
"percentage_new_agents_iteration": 0.35,
```
