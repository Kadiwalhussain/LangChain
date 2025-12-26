"""Command-line interface for Financial Assistant."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import questionary
from pathlib import Path
import logging
from typing import Optional

from src.document_processor import DocumentProcessor
from src.vector_store_manager import VectorStoreManager
from src.qa_chain import FinancialQAChain
from src.form_analyzer import FormAnalyzer
from src.deduction_finder import DeductionFinder
from config import settings

app = typer.Typer()
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@app.command()
def index_documents(directory: str = typer.Option(None, help="Directory to index")):
    """Index financial documents into vector store."""
    console.print("\n[bold blue]📚 Indexing Financial Documents[/bold blue]\n")
    
    if not directory:
        directory = questionary.path(
            "Enter directory path to index:",
            default=settings.knowledge_base_dir
        ).ask()
    
    if not Path(directory).exists():
        console.print(f"[red]❌ Directory not found: {directory}[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing documents...", total=None)
            
            # Process documents
            processor = DocumentProcessor()
            documents = processor.process_directory(directory)
            
            progress.update(task, description=f"[cyan]Found {len(documents)} document chunks")
            
            # Add to vector store
            vector_manager = VectorStoreManager()
            vector_manager.initialize_store()
            vector_manager.add_documents(documents)
            
            progress.update(task, completed=True)
        
        console.print(f"\n[green]✓ Successfully indexed {len(documents)} document chunks![/green]")
        
        # Show stats
        stats = vector_manager.get_collection_stats()
        console.print(f"Total documents in store: {stats['total_documents']}")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ask(question: Optional[str] = typer.Argument(None)):
    """Ask a question about your financial documents."""
    console.print("\n[bold blue]💬 Financial Q&A Assistant[/bold blue]\n")
    
    if not question:
        question = questionary.text(
            "What would you like to know about your finances?",
            multiline=False
        ).ask()
    
    if not question:
        console.print("[yellow]No question provided.[/yellow]")
        return
    
    try:
        with console.status("[bold green]Thinking..."):
            qa = FinancialQAChain()
            result = qa.ask(question)
        
        # Display answer
        console.print(Panel(
            result['answer'],
            title="[bold]Answer[/bold]",
            border_style="green"
        ))
        
        # Display sources
        if result['sources']:
            console.print("\n[bold]Sources:[/bold]")
            table = Table(show_header=True)
            table.add_column("File", style="cyan")
            table.add_column("Category", style="magenta")
            table.add_column("Page", style="yellow")
            
            for source in result['sources']:
                table.add_row(
                    source['file'],
                    source['category'],
                    str(source['page'])
                )
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze_form(file_path: str = typer.Argument(None, help="Path to tax form PDF")):
    """Analyze a tax form for completeness and issues."""
    console.print("\n[bold blue]🔍 Form Analyzer[/bold blue]\n")
    
    if not file_path:
        file_path = questionary.path(
            "Enter path to tax form (PDF):",
            only_files=True
        ).ask()
    
    if not Path(file_path).exists():
        console.print(f"[red]❌ File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    try:
        with console.status("[bold green]Analyzing form..."):
            analyzer = FormAnalyzer()
            result = analyzer.analyze_form(file_path)
        
        # Display results
        console.print(Panel(
            f"[bold]Form Type:[/bold] {result.get('form_type', 'Unknown')}\n"
            f"[bold]Completeness:[/bold] {result.get('completeness_score', 0)}%",
            title="[bold]Analysis Results[/bold]",
            border_style="blue"
        ))
        
        # Missing fields
        if result.get('missing_fields'):
            console.print("\n[bold red]Missing Fields:[/bold red]")
            for field in result['missing_fields']:
                console.print(f"  • {field}")
        
        # Warnings
        if result.get('warnings'):
            console.print("\n[bold yellow]⚠️  Warnings:[/bold yellow]")
            for warning in result['warnings']:
                console.print(f"  • {warning}")
        
        # Recommendations
        if result.get('recommendations'):
            console.print("\n[bold green]💡 Recommendations:[/bold green]")
            for rec in result['recommendations']:
                console.print(f"  • {rec}")
        
        # Summary
        console.print(f"\n[bold]Summary:[/bold] {result.get('summary', 'N/A')}")
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def find_deductions():
    """Find applicable tax deductions based on your profile."""
    console.print("\n[bold blue]💰 Deduction Finder[/bold blue]\n")
    
    # Gather user info
    profession = questionary.text(
        "What is your profession?",
        default="Software Engineer"
    ).ask()
    
    income = questionary.text(
        "What is your annual income?",
        default="100000"
    ).ask()
    
    expense_choices = questionary.checkbox(
        "Select your expense categories:",
        choices=[
            "home_office",
            "computer",
            "software",
            "internet",
            "phone",
            "education",
            "health_insurance",
            "travel",
            "meals"
        ]
    ).ask()
    
    filing_status = questionary.select(
        "Filing status:",
        choices=["single", "married_filing_jointly", "married_filing_separately", "head_of_household"]
    ).ask()
    
    try:
        with console.status("[bold green]Finding applicable deductions..."):
            finder = DeductionFinder()
            deductions = finder.find_deductions(
                profession=profession,
                expenses=expense_choices,
                income=float(income),
                filing_status=filing_status
            )
        
        # Display deductions
        console.print(f"\n[bold green]Found {len(deductions)} applicable deductions:[/bold green]\n")
        
        for i, d in enumerate(deductions, 1):
            console.print(Panel(
                f"[bold]{d['name']}[/bold]\n\n"
                f"[cyan]Category:[/cyan] {d['category']}\n"
                f"[cyan]Description:[/cyan] {d['description']}\n\n"
                f"[yellow]Estimated Savings:[/yellow] ${d.get('estimated_savings', 0):,.2f}\n\n"
                f"[green]Documentation Required:[/green]\n"
                + "\n".join(f"  • {doc}" for doc in d['documentation_required'][:3]) + "\n\n"
                f"[blue]💡 Tip:[/blue] {d['tips']}",
                title=f"[bold]Deduction #{i}[/bold]",
                border_style="green"
            ))
        
        # Total savings
        total_savings = finder.estimate_total_savings(deductions, tax_rate=0.22)
        console.print(f"\n[bold green]💵 Estimated Total Tax Savings: ${total_savings:,.2f}[/bold green]")
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats():
    """Show vector store statistics."""
    console.print("\n[bold blue]📊 Vector Store Statistics[/bold blue]\n")
    
    try:
        vector_manager = VectorStoreManager()
        vector_manager.initialize_store()
        stats = vector_manager.get_collection_stats()
        
        table = Table(title="Collection Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Documents", str(stats.get('total_documents', 0)))
        table.add_row("Collection Name", stats.get('collection_name', 'N/A'))
        table.add_row("Embedding Model", stats.get('embedding_model', 'N/A'))
        table.add_row("Vector Store Path", settings.vector_store_path)
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def interactive():
    """Start interactive Q&A session."""
    console.print(Panel(
        "[bold]Welcome to Financial Assistant![/bold]\n\n"
        "Ask questions about your tax documents, forms, and finances.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="[bold blue]💰 Financial Assistant[/bold blue]",
        border_style="blue"
    ))
    
    qa = FinancialQAChain()
    
    while True:
        question = questionary.text("\nYour question:").ask()
        
        if not question or question.lower() in ['exit', 'quit', 'q']:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        
        try:
            with console.status("[bold green]Thinking..."):
                result = qa.ask(question)
            
            console.print(Panel(
                result['answer'],
                title="[bold]Answer[/bold]",
                border_style="green"
            ))
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    app()
