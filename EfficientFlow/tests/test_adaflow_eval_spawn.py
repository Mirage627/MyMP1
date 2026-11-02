"""Exercise evaluator package registration with real stdlib spawn workers.

These probes test deployment imports only. They neither load a policy nor
produce simulated experiment metrics.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts" / "efficientadaflow_experiment.py"
PROBE_VALUE = "adaflow-spawn-import-ok"

WORKER_SOURCE = textwrap.dedent('''\
    import os
    from pathlib import Path
    import sys

    def deployment_probe(connection):
        package = sys.modules["EfficientFlow"]
        connection.send({
            "value": "adaflow-spawn-import-ok",
            "worker_module": __name__,
            "worker_file": str(Path(__file__).resolve()),
            "registered_source": getattr(package, "_efficientadaflow_source_root", None),
            "main_name": sys.modules["__main__"].__name__,
            "child_pid": os.getpid(),
        })
        connection.close()
''')

PROBE_MAIN = textwrap.dedent('''\

    if __name__ == "__main__":
        import multiprocessing
        import os
        from EfficientFlow.worker import deployment_probe

        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=deployment_probe, args=(send,))
        process.start()
        send.close()
        try:
            if not receive.poll(10):
                raise RuntimeError("Spawned deployment probe did not return a result")
            result = receive.recv()
            process.join(10)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("Spawned deployment probe failed: " + str(process.exitcode))
            result["parent_pid"] = os.getpid()
            print(json.dumps(result), flush=True)
        finally:
            receive.close()
            if process.is_alive():
                process.terminate()
                process.join(5)
            process.close()
''')


class EvaluatorSpawnDeploymentTests(unittest.TestCase):
    def _exercise_layout(self, nested):
        # Keep temporary files inside the explicitly writable repository, rather
        # than relying on Windows sandbox-specific system Temp permissions.
        with tempfile.TemporaryDirectory(prefix=".adaflow-spawn-test-", dir=ROOT) as temporary:
            base = Path(temporary).resolve()
            self.assertEqual(base.parent, ROOT)
            project = base / "deployment-with-an-arbitrary-name"
            source = project / "EfficientFlow" if nested else project
            launcher = project / "scripts" / "efficientadaflow_experiment.py"
            launcher.parent.mkdir(parents=True)
            for directory in (source / "policy", source / "common"):
                directory.mkdir(parents=True, exist_ok=True)
            (source / "__init__.py").write_text("# Import-only deployment fixture.\n", encoding="utf-8")
            (source / "policy" / "EfficientAdaFlow.py").write_text("", encoding="utf-8")
            (source / "common" / "adaptive_flow.py").write_text("", encoding="utf-8")
            (source / "worker.py").write_text(WORKER_SOURCE, encoding="utf-8")

            original = ENTRY.read_text(encoding="utf-8")
            main_guard = '\nif __name__ == "__main__":\n'
            self.assertIn(main_guard, original)
            # Preserve the actual evaluator's complete top-level execution;
            # replace only its command dispatch with the multiprocessing probe.
            launcher.write_text(original.rsplit(main_guard, 1)[0] + PROBE_MAIN, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", str(launcher)],
                cwd=base,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0,
                msg=f"Deployment probe failed.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            messages = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
            self.assertEqual(len(messages), 1, completed.stdout)
            result = messages[0]
            self.assertEqual(result["value"], PROBE_VALUE)
            self.assertEqual(result["worker_module"], "EfficientFlow.worker")
            self.assertEqual(Path(result["worker_file"]), (source / "worker.py").resolve())
            self.assertEqual(result["registered_source"], str(source.resolve()))
            self.assertEqual(result["main_name"], "__mp_main__")
            self.assertNotEqual(result["child_pid"], result["parent_pid"])

    def test_flat_source_with_arbitrary_directory_name(self):
        self._exercise_layout(nested=False)

    def test_outer_launcher_with_nested_efficientflow_package(self):
        self._exercise_layout(nested=True)


if __name__ == "__main__":
    unittest.main()
