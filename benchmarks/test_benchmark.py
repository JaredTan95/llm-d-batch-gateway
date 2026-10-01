import unittest
from unittest import mock

from benchmarks import benchmark


class FlowControlMetricsTest(unittest.TestCase):
    @mock.patch.object(benchmark, "query_prometheus")
    def test_current_family_is_used_when_both_families_exist_in_namespace(self, query):
        target_namespace = "batch-bench-s4"
        namespace_selector = f'namespace="{target_namespace}"'
        current_queries = [
            f"avg(llm_d_epp_flow_control_pool_saturation{{{namespace_selector}}})",
            f"sum by (priority) (llm_d_epp_flow_control_queue_size{{{namespace_selector}}})",
            f"sum(llm_d_epp_flow_control_queue_size{{{namespace_selector}}})",
        ]
        legacy_queries = [
            f"avg(inference_extension_flow_control_pool_saturation{{{namespace_selector}}})",
            f"sum by (priority) (inference_extension_flow_control_queue_size{{{namespace_selector}}})",
            f"sum(inference_extension_flow_control_queue_size{{{namespace_selector}}})",
        ]
        available_results = {
            current_queries[0]: [{"values": [[0, "0.25"], [15, "0.5"]]}],
            current_queries[1]: [
                {"metric": {"priority": "-1"}, "values": [[0, "3"], [15, "5"]]}
            ],
            current_queries[2]: [{"values": [[0, "4"], [15, "6"]]}],
            legacy_queries[0]: [{"values": [[0, "0.9"]]}],
            legacy_queries[1]: [
                {"metric": {"priority": "-1"}, "values": [[0, "90"]]}
            ],
            legacy_queries[2]: [{"values": [[0, "90"]]}],
        }

        def query_result(_context, _namespace, promql, _start_time, _end_time):
            return available_results[promql]

        query.side_effect = query_result

        metrics = benchmark.collect_flow_control_metrics(
            "test-context", target_namespace, mock.sentinel.start, mock.sentinel.end
        )

        self.assertEqual(0.375, metrics["flow_control_saturation_avg"])
        self.assertEqual(4, metrics["queue_size_priority_-1_avg"])
        self.assertEqual(5, metrics["flow_control_queue_size_avg"])
        self.assertEqual(current_queries, [call.args[2] for call in query.call_args_list])

    @mock.patch.object(benchmark, "query_prometheus")
    def test_legacy_family_is_selected_with_new_family_in_another_namespace(self, query):
        target_namespace = "batch-bench-s4"
        namespace_selector = f'namespace="{target_namespace}"'
        queries = []

        def query_result(_context, _namespace, promql, _start_time, _end_time):
            queries.append(promql)
            self.assertIn(namespace_selector, promql)

            # The shared Prometheus also has llm_d_epp metrics, but only in a
            # different namespace, so the scoped current-family probe is empty.
            if promql.startswith("avg(llm_d_epp_"):
                return []
            if promql.startswith("avg(inference_extension_"):
                return [{"values": [[0, "0.5"], [15, "0.75"]]}]
            if promql.startswith("sum by (priority)"):
                return [{"metric": {"priority": "-1"}, "values": [[0, "2"]]}]
            if promql.startswith("sum(inference_extension_"):
                return [{"values": [[0, "2"], [15, "3"]]}]
            self.fail(f"Unexpected query: {promql}")

        query.side_effect = query_result

        metrics = benchmark.collect_flow_control_metrics(
            "test-context", target_namespace, mock.sentinel.start, mock.sentinel.end
        )

        self.assertEqual(0.625, metrics["flow_control_saturation_avg"])
        self.assertEqual(2.5, metrics["flow_control_queue_size_avg"])
        self.assertEqual(
            [
                'avg(llm_d_epp_flow_control_pool_saturation{namespace="batch-bench-s4"})',
                'avg(inference_extension_flow_control_pool_saturation{namespace="batch-bench-s4"})',
                'sum by (priority) (inference_extension_flow_control_queue_size{namespace="batch-bench-s4"})',
                'sum(inference_extension_flow_control_queue_size{namespace="batch-bench-s4"})',
            ],
            queries,
        )


if __name__ == "__main__":
    unittest.main()
