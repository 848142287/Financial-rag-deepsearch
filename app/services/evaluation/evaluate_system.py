#!/usr/bin/env python3
"""
金融RAG系统评测脚本
使用生成的数据集评测系统性能
"""

import json
import requests
import time
from typing import Dict, List, Tuple
import statistics
from datetime import datetime

class SystemEvaluator:
    def __init__(self, base_url: str = "http://localhost:3014"):
        self.base_url = base_url
        self.results = []

    def load_dataset(self, dataset_file: str) -> List[Dict]:
        """加载评测数据集"""
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset['questions']

    def evaluate_single_question(self, question: Dict) -> Dict:
        """评测单个问题"""
        start_time = time.time()

        try:
            # 发送问题到RAG系统
            response = requests.post(
                f"{self.base_url}/api/v1/rag/stream-query",
                json={
                    "query": question['question'],
                    "conversation_id": "evaluation_session"
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "question_id": question['id'],
                    "question": question['question'],
                    "expected_complexity": question['complexity'],
                    "expected_difficulty": question['difficulty_score'],
                    "status": "success",
                    "response_time": response_time,
                    "answer": result.get('answer', ''),
                    "sources": result.get('sources', []),
                    "confidence": result.get('confidence', 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "question_id": question['id'],
                    "question": question['question'],
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            return {
                "question_id": question['id'],
                "question": question['question'],
                "status": "error",
                "error": str(e),
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

    def evaluate_subset(self, questions: List[Dict], sample_size: int = 50) -> List[Dict]:
        """评测问题的子集"""
        # 随机选择问题进行评测
        sample_questions = questions[:sample_size] if len(questions) <= sample_size else \
                          questions[:sample_size]  # 取前50个问题作为示例

        print(f"开始评测 {len(sample_questions)} 个问题...")

        results = []
        for i, question in enumerate(sample_questions, 1):
            print(f"评测进度: {i}/{len(sample_questions)} - 问题: {question['question'][:50]}...")
            result = self.evaluate_single_question(question)
            results.append(result)

            # 避免请求过快
            time.sleep(0.5)

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """分析评测结果"""
        success_results = [r for r in results if r['status'] == 'success']
        error_results = [r for r in results if r['status'] == 'error']

        if not success_results:
            return {
                "total_questions": len(results),
                "successful_questions": 0,
                "error_questions": len(error_results),
                "success_rate": 0,
                "error_analysis": error_results
            }

        response_times = [r['response_time'] for r in success_results]

        # 按复杂度分析
        complexity_stats = {}
        for complexity in ['simple', 'medium', 'complex']:
            complexity_results = [r for r in success_results
                                if r.get('expected_complexity') == complexity]
            if complexity_results:
                complexity_times = [r['response_time'] for r in complexity_results]
                complexity_stats[complexity] = {
                    "count": len(complexity_results),
                    "avg_response_time": statistics.mean(complexity_times),
                    "min_response_time": min(complexity_times),
                    "max_response_time": max(complexity_times),
                    "avg_confidence": statistics.mean([r.get('confidence', 0) for r in complexity_results])
                }

        return {
            "total_questions": len(results),
            "successful_questions": len(success_results),
            "error_questions": len(error_results),
            "success_rate": len(success_results) / len(results) * 100,
            "overall_stats": {
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "median_response_time": statistics.median(response_times),
                "avg_confidence": statistics.mean([r.get('confidence', 0) for r in success_results])
            },
            "complexity_analysis": complexity_stats,
            "error_analysis": error_results[:5],  # 只显示前5个错误
            "timestamp": datetime.now().isoformat()
        }

    def save_results(self, results: List[Dict], analysis: Dict, output_file: str):
        """保存评测结果"""
        report = {
            "evaluation_metadata": {
                "base_url": self.base_url,
                "evaluation_date": datetime.now().isoformat(),
                "total_evaluated": len(results)
            },
            "analysis": analysis,
            "detailed_results": results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

def main():
    # 创建评测器
    evaluator = SystemEvaluator()

    # 加载数据集
    print("加载评测数据集...")
    questions = evaluator.load_dataset("dataset_evaluation.json")
    print(f"加载了 {len(questions)} 个问题")

    # 评测子集（作为示例）
    results = evaluator.evaluate_subset(questions, sample_size=20)

    # 分析结果
    analysis = evaluator.analyze_results(results)

    # 保存结果
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.save_results(results, analysis, output_file)

    # 打印报告
    print("\n=== 系统评测报告 ===")
    print(f"评测问题总数: {analysis['total_questions']}")
    print(f"成功回答: {analysis['successful_questions']}")
    print(f"失败回答: {analysis['error_questions']}")
    print(f"成功率: {analysis['success_rate']:.2f}%")

    if analysis['overall_stats']:
        print(f"\n响应时间统计:")
        print(f"  平均响应时间: {analysis['overall_stats']['avg_response_time']:.2f}秒")
        print(f"  最快响应时间: {analysis['overall_stats']['min_response_time']:.2f}秒")
        print(f"  最慢响应时间: {analysis['overall_stats']['max_response_time']:.2f}秒")
        print(f"  中位数响应时间: {analysis['overall_stats']['median_response_time']:.2f}秒")
        print(f"  平均置信度: {analysis['overall_stats']['avg_confidence']:.2f}")

    print(f"\n复杂度分析:")
    for complexity, stats in analysis['complexity_analysis'].items():
        print(f"  {complexity}: {stats['count']}个问题, "
              f"平均响应时间{stats['avg_response_time']:.2f}秒, "
              f"平均置信度{stats['avg_confidence']:.2f}")

    if analysis['error_analysis']:
        print(f"\n错误示例 (前5个):")
        for error in analysis['error_analysis']:
            print(f"  - {error['question_id']}: {error.get('error', 'Unknown error')}")

    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()