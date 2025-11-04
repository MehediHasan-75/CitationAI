from fastapi import APIRouter

router = APIRouter(
    prefix= '/api',
    tags= ['Query History & Analytics']
)

@router.get('/queries/history')
def queries():
    return "This is all histories"

@router.get('/analytics/popular')
def popular():
    return "This is all popular"