# EvoProspect

## 1. Raw data:

The monkey data is split in two files: 
 - monkeys/source/data.xlsx
 - monkeys/source/data_GH.xlsx
 
Note: Analysis takes into account only results 
after:
 * 2017-03-01 for 'data_GH.xlsx';
 * 2020-02-18 for 'data.xlsx'

## 2. Monkeys

### Reproduce figures
    
#### Config

* Current directory has to be 'monkeys'
        
        cd monkeys

* Create empty database using Django ORM

        python3 manage.py makemigrations
        python3 manage.py migrate


* Import the data

    python3 import_xlsx
    

#### Run 
    python3 main.py
    

### Data viewing
   
#### Config

* Create a superuser
        
        python3 manage.py createsuperuser

* Launch the server
    
       python3 manage.py runserver 
 
 #### Use

Go to http://127.0.0.1:8000/
    
    
## 3. Simulations

### Reproduce figures

#### Config

Current directory has to be 'simulation'

    cd simulation
    
#### Run

    python3 gain-analysis.py
    python3 agent-simulation.py
    python3 monkey-agent-comparison.py
    