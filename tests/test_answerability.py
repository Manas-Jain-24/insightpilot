import unittest

from backend.agent.answerability import answer_parts, missing_fact, needs_subject
from backend.agent.gemini_client import GeminiClient
from backend.models import Citation


def document(id, title, snippet):
    return Citation(id=id, source_type="document", title=title, doc_id=str(id), snippet=snippet)


class AnswerabilityTests(unittest.TestCase):
    def test_only_explicit_question_clauses_are_split(self):
        self.assertEqual(answer_parts("How long is a telescope reservation and when can I cancel it?"),
                         ["How long is a telescope reservation", "when can I cancel it"])
        self.assertEqual(answer_parts("Compare cats and dogs"), ["Compare cats and dogs"])

    def test_explicit_subject_is_not_lost_because_a_later_clause_says_it(self):
        source = document(1, "Telescope reservations", "A telescope reservation lasts two hours. Members may book a telescope five days ahead.")
        query = "How long is a telescope reservation and how far ahead can members book it?"
        self.assertFalse(needs_subject(query, [source]))
        result = GeminiClient(api_key="")._generate_fallback_report(query, [], [source], "balanced")
        self.assertIn("two hours", result.executive_summary)
        self.assertIn("five days", result.executive_summary)
        self.assertEqual(len(result.key_findings), 2)

    def test_related_battery_text_does_not_supply_price_or_temperature(self):
        sources = [document(1, "Battery", "Charge the battery before departure.")]
        self.assertTrue(missing_fact("What is the battery price?", sources))
        self.assertTrue(missing_fact("What temperature can the battery withstand?", sources))
        self.assertFalse(missing_fact("How do I charge the battery?", sources))

    def test_missing_attribution_does_not_accept_an_unrelated_creation_fact(self):
        source = document(1, "Museum", "The museum creates a visitor badge on arrival.")
        self.assertTrue(missing_fact("Who founded the museum?", [source]))

    def test_ambiguous_scoped_values_require_a_subject(self):
        sources = [document(1, "Member invitation.txt", "An invitation link expires after eight hours."),
                   document(2, "Guest invitation.txt", "An invitation link expires after one hour.")]
        self.assertTrue(needs_subject("When does the invitation link expire?", sources))
        self.assertFalse(needs_subject("When does the member invitation link expire?", sources))
        result = GeminiClient(api_key="")._generate_fallback_report("When does the invitation link expire?", [], sources, "balanced")
        self.assertEqual(result.metrics["mode"], "clarification_needed")
        self.assertEqual(result.key_findings, [])

    def test_partial_answers_identify_the_unanswered_clause(self):
        source = document(1, "Telescope", "A telescope reservation lasts two hours.")
        result = GeminiClient(api_key="")._generate_fallback_report(
            "How long is a telescope reservation and where are spare lenses stored?", [], [source], "balanced")
        self.assertIn("two hours", result.executive_summary)
        self.assertEqual(result.metrics["unanswered_parts"], ["where are spare lenses stored"])


if __name__ == "__main__":
    unittest.main()
