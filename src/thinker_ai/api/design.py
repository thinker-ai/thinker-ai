import json
import os
from fastapi import APIRouter, Depends
from starlette.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.api.login import get_session
from thinker_ai.app.design.solution.solution_manager import SolutionManager
from thinker_ai.configs.const import PROJECT_ROOT

design_router = APIRouter()
design_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'design')
design_dir = Jinja2Templates(directory=design_root)
solution_manager = SolutionManager()


@design_router.get("/design", response_class=HTMLResponse)
async def main(request: Request):
    return design_dir.TemplateResponse("design.html", {"request": request})


@design_router.get("/design/list", response_class=HTMLResponse)
async def design_list(request: Request):
    return design_dir.TemplateResponse("list.html", {"request": request})


@design_router.get("/design/one", response_class=HTMLResponse)
async def design_one(request: Request):
    return design_dir.TemplateResponse("one.html", {"request": request})


@design_router.get("/design/one/criterion", response_class=HTMLResponse)
async def design_one_criterion(request: Request):
    return design_dir.TemplateResponse("one/criterion.html", {"request": request})


@design_router.get("/design/one/solution", response_class=HTMLResponse)
async def design_one_solution(request: Request):
    return design_dir.TemplateResponse("one/solution.html", {"request": request})


@design_router.get("/design/one/strategy", response_class=HTMLResponse)
async def design_one_strategy(request: Request):
    return design_dir.TemplateResponse("one/strategy.html", {"request": request})


@design_router.get("/design/one/resources", response_class=HTMLResponse)
async def design_one_resource(request: Request):
    return design_dir.TemplateResponse("one/resources.html", {"request": request})


@design_router.get("/design/one/resources/tools", response_class=HTMLResponse)
async def design_one_resources_tools(request: Request):
    return design_dir.TemplateResponse("one/resources/tools.html", {"request": request})


@design_router.get("/design/one/resources/solutions", response_class=HTMLResponse)
async def design_one_resources_solutions(request: Request):
    return design_dir.TemplateResponse("one/resources/solutions.html", {"request": request})


@design_router.get("/design/one/resources/trains", response_class=HTMLResponse)
async def design_one_resources_trains(request: Request):
    return design_dir.TemplateResponse("one/resources/trains.html", {"request": request})


@design_router.get("/design/one/resources/data_sources", response_class=HTMLResponse)
async def design_one_resources_data_sources(request: Request):
    return design_dir.TemplateResponse("one/resources/data_sources.html", {"request": request})


@design_router.get("/design/one/resources/third_parties", response_class=HTMLResponse)
async def design_one_resources_third_party(request: Request):
    return design_dir.TemplateResponse("one/resources/third_parties.html", {"request": request})


@design_router.post("/design/one/solution/generate_state_machine_def", response_class=JSONResponse)
async def design_one_solution_generate_state_machine_def(request: Request,
                                                         session: dict = Depends(get_session)) -> dict:
    user_id = session.get("user_id")
    solution = solution_manager.get_not_done(user_id)
    if solution:
        # 获取请求体中的 JSON 数据
        body = await request.json()
        name = body.get("name")
        description = body.get("description")
        is_root = body.get("is_root")
        if is_root:
            solution.name = name
            solution.description = description
        await solution.generate_state_machine_definition(name, description)
        solution_manager.save(solution)
        return await solution.to_dict_include_menu_tree()
    else:
        return {}


@design_router.get("/design/one/solution/current", response_class=JSONResponse)
async def design_one_solution_current(session: dict = Depends(get_session)) -> dict:
    user_id = session.get("user_id")
    solution = solution_manager.get_not_done(user_id)
    solution_dict = {}
    if solution:
        solution_dict = await solution.to_dict_include_menu_tree()
    return solution_dict
    # return {
    #     "name": "solution.name",
    #     "description": "solution.description",
    #     "solution_tree":[
    #     {
    #         "name": "问题分解 1",
    #         "description": "详情内容 1",
    #         "children": [
    #             {
    #                 "name": "叶子节点 1.1",
    #                 "description": "叶子节点 1.1 的详细内容",
    #                 "children": []
    #             },
    #             {
    #                 "name": "叶子节点 1.2",
    #                 "description": "叶子节点 1.2 的详细内容",
    #                 "children": []
    #             }
    #         ]
    #     },
    #     {
    #         "name": "问题分解 2",
    #         "description": "详情内容 2",
    #         "children": [
    #             {
    #                 "name": "问题分解 2.1",
    #                 "description": "问题分解 2.1 的详细内容",
    #                 "children": [
    #                     {
    #                         "name": "叶子节点 2.1.1",
    #                         "description": "叶子节点 2.1.1 的详细内容",
    #                         "children": []
    #                     },
    #                     {
    #                         "name": "叶子节点 2.1.2",
    #                         "description": "叶子节点 2.1.2 的详细内容",
    #                         "children": []
    #                     }
    #                 ]
    #             },
    #             {
    #                 "name": "叶子节点 2.2",
    #                 "description": "叶子节点 2.2 的详细内容",
    #                 "children": []
    #             }
    #         ]
    #     }
    # ]
    # }
