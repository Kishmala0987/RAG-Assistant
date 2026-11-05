#This is a python image
FROM python:3.12.0-slim

# we have created virtual env and used uv for it
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/ 

#create app directory
WORKDIR /app

#pull the requirements
COPY requirements.txt .

#Dont pull any cache from pycache and directly run the requirements
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu


RUN uv pip install --system --no-cache -r requirements.txt

#copy my code
#will copy whole workbase of 'PROJECT' but we donot need to copy everythong so we willcreate a .dockerignore
COPY . .

#expose wheere the app will run PORT MAPPING
EXPOSE 8501

#how to run app
CMD ["streamlit","run", "app.py", "--server.address=0.0.0.0"]