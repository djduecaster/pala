# Behavior memory boundary

The current hold-only runtime has no model memory or scene interpretation.
Earlier identity/transcript experiments are historical and are not loaded by
`pala.main`.

The next behavior slice will use a small amount of execution context: active
skill, status, recent accepted outcomes, and greeting/re-arm bookkeeping.
This contract remains to be designed with the skill runner. Persistent
autobiographical memory and a separate summarizer are outside the V1 scope.
