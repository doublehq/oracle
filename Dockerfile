FROM public.ecr.aws/lambda/python:3.13 AS build

RUN dnf install -y make git wget tar patch dos2unix pkg-config gcc gcc-c++ gcc-gfortran
RUN wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
RUN chmod u+x coinbrew
RUN ./coinbrew fetch Cbc@2.10.12 --no-third-party
RUN ./coinbrew build Cbc

FROM public.ecr.aws/lambda/python:3.13 AS runtime
COPY --from=build ${LAMBDA_TASK_ROOT}/dist ${LAMBDA_TASK_ROOT}/dist
ENV PATH="${PATH}:${LAMBDA_TASK_ROOT}/dist/bin"

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

COPY solvers/ ${LAMBDA_TASK_ROOT}/solvers/
COPY service/ ${LAMBDA_TASK_ROOT}/service/

COPY lambda_function.py ${LAMBDA_TASK_ROOT}
CMD ["lambda_function.lambda_handler"]

ARG VERSION="unknown"
ENV VERSION=${VERSION}
