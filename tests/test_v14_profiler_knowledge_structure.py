import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stdout


class ProfilerKnowledgeStructureBlackBoxTests(unittest.TestCase):
    def test_empty_knowledge_match_is_a_successful_bounded_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            completed = project.run_tool(
                "knowledge_query.py",
                "query",
                project.knowledge_input(),
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"query failed:\nstdout={completed.stdout}\nstderr={completed.stderr}",
            )
            result = decode_stdout(completed)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["matches"], [])
            self.assertLessEqual(result["context_bytes"], 4096)
            self.assertFalse(project.artifact_root.exists())


if __name__ == "__main__":
    unittest.main()
