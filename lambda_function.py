from src.service.oracle import Oracle

def lambda_handler(event, context):
    return {
        "lambda": {
            "function_version": (context.function_version if context else None),
            "aws_request_id": (context.aws_request_id if context else None),
        },
        "oracle": Oracle.process_lambda_event(event),
    }
