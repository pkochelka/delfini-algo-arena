"""Strategy results explorer: metrics and an interactive HTML report over one or more simulator runs."""

from results_explorer.load import Run, load_run
from results_explorer.metrics import Analysis, analyse

__all__ = ['Analysis', 'Run', 'analyse', 'load_run']
