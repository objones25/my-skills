# 02: Is the email-triage agent safe against prompt injection?

*Illustrative system.* One agent reads inbound customer email, looks up accounts in a CRM, and sends replies on its own.

## Verdict

Not as designed. One session holds untrusted input (inbound email), private data (CRM, and the inbox itself), and an outbound channel (sending mail). That is the combination Meta's Rule of Two says a session should never hold without a human or other reliable check on the combining step. Filters won't close it: published injection defenses fall to adaptive attacks (`security.md`).

## The audit

Describe tools and sessions in a manifest (`tests/fixtures/email_agent_manifest.json`) and run:

```
$ python scripts/rule_of_two_audit.py tests/fixtures/email_agent_manifest.json
single_agent: holds ABC  VIOLATION
  A via read_inbox
  B via read_inbox, crm_lookup
  C via send_reply
  fix: gate send_reply with human approval, or split so no session holds A, B and C; ...
gated_agent: holds ABC
  ...
reader: holds AB
  ...
sender_free_text: holds BC  VIOLATION
  ...
  A inherited (free-text input) from reader
sender_structured: holds BC
  ...
```

Three designs pass and two fail:

- `gated_agent` keeps all three properties but puts a human on `send_reply`. Safe, but watch approval fatigue; if reviewers approve nearly everything, the gate is decoration.
- `reader` + `sender_structured` splits the work. The reader sees email but can't send. The sender can send but receives only a fixed-schema result (intent label, account id, template id), not free text it might obey.
- `sender_free_text` is the common mistake: the split exists on paper, but the reader passes a free-text summary, so the injection rides along. The audit counts that as inherited [A].

## Channels people forget

Count as [C] anything that can carry data out: a rendered image URL with a query string, a link in a reply, a parameter on an API call. "It can only draft, not send" isn't enough if drafts render remote images.

## Test it

Add injection cases to the eval: an email asking the agent to forward account details elsewhere, one hiding instructions in quoted text, one with a tracking-pixel URL. The pass condition is an outcome check (no outbound message or URL contains CRM data), not the model saying it refused.
