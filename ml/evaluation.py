from typing import Dict, Union                                                                                                                       
import numpy as np                                                                                                                                   
import pandas as pd                                                                                                                                  
from sklearn.metrics import mean_squared_error, r2_score, f1_score, classification_report, confusion_matrix                                          
from rich.console import Console                                                                                                                     
from rich.table import Table                                                                                                                         
from rich.panel import Panel                                                                                                                         
                                                                                                                                                    
console = Console()                                                                                                                                  
                                                                                                                                                    
class Evaluator:                                                                                                                                     
    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "",
        is_classification: bool = False, class_names: list[str] | None = None,
    ) -> Dict[str, Union[float, str]]:                                                                                                               
        if is_classification:                                                                                                                        
            f1 = f1_score(y_true, y_pred, average="macro")                                                                                           
            # Get report as a dictionary for precise mapping                                                                                         
            report_dict = classification_report(y_true, y_pred, output_dict=True)                                                                    
            cm = confusion_matrix(y_true, y_pred)                                                                                                    
                                                                                                                                                    
            title = f"[bold cyan]{model_name}[/bold cyan]" if model_name else "[bold cyan]Classification Results[/bold cyan]"                        
                                                                                                                                                    
            # 1. Summary Panel                                                                                                                       
            summary_content = f"[bold white]Macro-F1 Score: [green]{f1:.4f}[/green][/bold white]"
            console.print(Panel(summary_content, title=title, expand=False))                                                                         
                                                                                                                                                    
            # 2. Metrics Table - Mapping dict keys to columns                                                                                        
            metrics_table = Table(title="Detailed Classification Metrics", show_header=True, header_style="bold cyan")                               
            metrics_table.add_column("Class", style="bold")                                                                                          
            metrics_table.add_column("Precision", justify="right")                                                                                   
            metrics_table.add_column("Recall", justify="right")                                                                                      
            metrics_table.add_column("F1-Score", justify="right")                                                                                    
            metrics_table.add_column("Support", justify="right")                                                                                     
                                                                                                                                                    
            # Add per-class metrics                                                                                                                  
            for label, metrics in report_dict.items():                                                                                               
                if label == 'accuracy': continue                                                                                                     
                if isinstance(metrics, dict):                                                                                                        
                    # Handle 'macro avg', 'weighted avg', and numeric class labels                                                                   
                    label_str = str(label)                                                                                                           
                    metrics_table.add_row(                                                                                                           
                        label_str,                                                                                                                   
                        f"{metrics['precision']:.2f}",                                                                                               
                        f"{metrics['recall']:.2f}",                                                                                                  
                        f"{metrics['f1-score']:.2f}",                                                                                                
                        f"{int(metrics['support'])}"                                                                                                 
                    )                                                                                                                                
                                                                                                                                                    
            # Add Accuracy as a separate footer row                                                                                                  
            acc = report_dict.get('accuracy', 0)                                                                                                     
            metrics_table.add_section()                                                                                                              
            metrics_table.add_row("OVERALL ACCURACY", f"{acc:.2f}", "-", "-", "-")                                                                   
                                                                                                                                                    
            console.print(metrics_table)                                                                                                             
                                                                                                                                                    
            # 3. Confusion Matrix Table
            labels = class_names or [str(i) for i in range(len(cm))]
            cm_table = Table(title="Confusion Matrix (Actual ↓ | Predicted →)", show_header=True, header_style="bold yellow")
            cm_table.add_column("", style="bold yellow")
            for label in labels:
                cm_table.add_column(label, justify="right")
            for i, row in enumerate(cm):
                cm_table.add_row(labels[i], *[f"[bold white]{val}[/bold white]" for val in row])                                                                
                                                                                                                                                    
            console.print(cm_table)                                                                                                                  
            console.print("\n")                                                                                                                      
                                                                                                                                                    
            return {"f1": f1, "report": report_dict, "cm": cm}                                                                                       
                                                                                                                                                    
        # Regression path                                                                                                                            
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5                                                                                                                            
        r2 = r2_score(y_true, y_pred)                                                                                                                
                                                                                                                                                    
        title = f"[bold magenta]{model_name}[/bold magenta]" if model_name else "[bold magenta]Regression Results[/bold magenta]"                    
        metrics = f"MSE: [bold]{mse:.2f}[/bold] | RMSE: [bold]{rmse:.2f}s[/bold] | R²: [bold]{r2:.4f}[/bold]"                                        
                                                                                                                                                    
        console.print(Panel(metrics, title=title, expand=False))                                                                                     
        return {"mse": mse, "rmse": rmse, "r2": r2}