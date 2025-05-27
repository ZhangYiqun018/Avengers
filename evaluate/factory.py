from enum import Enum

from evaluate.AIME import AIMEEvaluator
from evaluate.GPQA import GPQAEvaluator
from evaluate.MATH500 import MATH500Evaluator
from evaluate.MedQA import MedQAEvaluator
from evaluate.MMLUPro import MMLUProEvaluator
from evaluate.EmoryNLP import EmoryNLPEvaluator
from evaluate.HumanEval import HumanEvalEvaluator
from evaluate.K_and_K import KnightsAndKnavesEvaluator
from evaluate.FinQA import FinQAEvaluator
from evaluate.MBPP import MBPPEvaluator
from evaluate.ARCC import ARCCEvaluator
from evaluate.Winogrande import WinograndeEvaluator
from evaluate.BBH import BBHEvaluator
from evaluate.MATHBench import MathBenchEvaluator
from evaluate.LiveMathBench import LiveMathBenchEvaluator
from evaluate.MELD import MELDEvaluator
from evaluate.LiveCodeBench import LiveCodeBenchEvaluator
from evaluate.KORBench import KORBenchEvaluator
from evaluate.ArenaHard import ArenaHardEvaluator
from evaluate.TruthfulQA import TruthfulQAEvaluator
from evaluate.DailyDialog import DailyDialogEvaluator
from evaluate.StudentEval import StudentEvalEvaluator
from evaluate.BrainTeaser import BrainTeaserEvaluator

class Benchmark(Enum):
    # math
    AIMETOTAL = 'aime_total'
    AIME2024 = 'aime2024'
    AIME2025 = 'aime2025'
    AIME = 'aime'
    MATH500 = 'math500'
    LIVEMATHBENCH = 'livemathbench'
    # mmlu
    MMLUPro = 'mmlupro'
    # emotion
    EmoryNLP = 'emorynlp'
    MELD = 'meld'
    # code
    HumanEval = 'humaneval'
    MBPP = 'mbpp'
    # logical
    KnightsAndKnaves = 'kandk'
    BBH = 'bbh'
    KORBench = 'korbench'
    # QA
    FinQA = 'finqa'
    MedQA = 'medqa'
    GPQA = 'gpqa'
    ARCC = 'arcc'
    SimpleQA = 'simpleqa'
    # Out of distribution
    TruthfulQA = 'truthfulqa' # knowledge
    MATHBENCH = 'mathbench'  # math
    LiveCodeBench = 'livecodebench' # code
    Winogrande = 'winogrande' # logic
    DailyDialog = 'dailydialog' # Affective Computing
    StudentEval = 'studenteval' # code
    BrainTeaser = 'brainteaser' # logic
    # arenahard
    ArenaHard = 'arenahard'
    
class EvaluatorFactory:
    def __init__(self, max_workers: int=8, mode: str="test"):
        self.max_workers = max_workers
        assert mode in ["test", "full"], f"Invalid mode: {mode}, mode should be in ['test', 'full']"
        self.mode = mode
    
    def get_evaluator(self, task: str | Benchmark):
        if isinstance(task, str):
            task = Benchmark(task)
        
        if not isinstance(task, Benchmark):
            raise TypeError(f"Invalid task type: {type(task)}, task: {task}")
        
        # AIME
        if task == Benchmark.AIME:
            return AIMEEvaluator(split='hybrid', max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.AIME2024:
            return AIMEEvaluator(split='2024', max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.AIME2025:
            return AIMEEvaluator(split='2025', max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.AIMETOTAL:
            return AIMEEvaluator(split='total', max_workers=self.max_workers, mode=self.mode)
        # MATH
        elif task == Benchmark.MATH500:
            return MATH500Evaluator(max_workers=self.max_workers, mode=self.mode)
        # MATHBENCH
        elif task == Benchmark.MATHBENCH:
            return MathBenchEvaluator(max_workers=self.max_workers, mode=self.mode)
        # LIVEMATHBENCH
        elif task == Benchmark.LIVEMATHBENCH:
            return LiveMathBenchEvaluator(max_workers=self.max_workers, mode=self.mode)
        # MMLUPro
        elif task == Benchmark.MMLUPro:
            return MMLUProEvaluator(split="test", max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.MedQA:
            return MedQAEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.GPQA:
            return GPQAEvaluator(max_workers=self.max_workers, mode=self.mode)
        # Affective Computing
        elif task == Benchmark.EmoryNLP:
            return EmoryNLPEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.MELD:
            return MELDEvaluator(max_workers=self.max_workers, mode=self.mode)
        # Code Generation
        elif task == Benchmark.HumanEval:
            return HumanEvalEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.MBPP:
            return MBPPEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.LiveCodeBench:
            return LiveCodeBenchEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.StudentEval:
            return StudentEvalEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.KnightsAndKnaves:
            return KnightsAndKnavesEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.BBH:
            return BBHEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.KORBench:
            return KORBenchEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.FinQA:
            return FinQAEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.ARCC:
            return ARCCEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.Winogrande:
            return WinograndeEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.TruthfulQA:
            return TruthfulQAEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.ArenaHard:
            return ArenaHardEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.DailyDialog:
            return DailyDialogEvaluator(max_workers=self.max_workers, mode=self.mode)
        elif task == Benchmark.BrainTeaser:
            return BrainTeaserEvaluator(max_workers=self.max_workers, mode=self.mode)
        else:
            raise ValueError(f"Invalid task: {task}")
        