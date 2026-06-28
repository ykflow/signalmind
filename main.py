import os
from pathlib import Path
import numpy as np
import pandas as pd
from workflows.workflows import MainSignalMindWorkflow



workflow = MainSignalMindWorkflow()
context = workflow.run()

