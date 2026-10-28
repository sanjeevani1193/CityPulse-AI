"""Schema → backend decision → validated execution. No implicit paid backend."""
from typing import Protocol
from citypulse_ai.sqlpilot.core import QueryRejected


class Backend(Protocol):
    def generate(self, question: str, schema: dict) -> dict:
        """Return {'sql': ...} or {'clarification': ...}; schema only, no sample rows."""


class SQLPilot:
    def __init__(self, database, backend: Backend | None = None):
        self.database = database; self.backend = backend

    def ask(self, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise QueryRejected('Provide a nonempty analytical question of at most 2000 characters')
        if self.backend is None:
            return {'status': 'backend_unavailable', 'question': question,
                    'explanation': 'No approved reachable LLM is configured. Schema and safe SQL tools work; no SQL or answer was invented.'}
        try:
            decision = self.backend.generate(question, self.database.discover()['schema'])
        except Exception:
            return {'status': 'backend_error', 'explanation': 'The configured LLM did not return a usable decision'}
        if not isinstance(decision, dict): raise QueryRejected('Invalid backend decision')
        if set(decision) == {'clarification'} and isinstance(decision['clarification'], str):
            return {'status': 'clarification', 'question': decision['clarification'][:1000]}
        if set(decision) != {'sql'}: raise QueryRejected('Backend must provide SQL or a clarification, not an ungrounded answer')
        return {'status': 'executed', **self.database.query(decision['sql'])}
