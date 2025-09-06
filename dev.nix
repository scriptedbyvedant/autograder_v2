      { pkgs, ... }: {
        packages = [
          pkgs.python311Packages.streamlit
          pkgs.python311Packages.reportlab
          pkgs.python311Packages.pymupdf
          pkgs.python311Packages.psycopg2
          pkgs.python311Packages.pandas
          pkgs.python311Packages.python-dotenv
          pkgs.python311Packages.langchain-core
          pkgs.python311Packages.langchain-community
          pkgs.python311Packages.langchain-ollama
          pkgs.python311Packages.langchain-groq
          pkgs.python311Packages.sympy
          pkgs.python311Packages.numpy
          pkgs.python311Packages.plotly
          pkgs.python311Packages.pytest
          pkgs.python311Packages.sentence-transformers
          pkgs.python311Packages.pip

          # For Fine-Tuning
          pkgs.python311Packages.torch
          pkgs.python311Packages.transformers
          pkgs.python311Packages.peft
          pkgs.python311Packages.datasets
          pkgs.python311Packages.bitsandbytes
          pkgs.python311Packages.accelerate
        ];
      } 
