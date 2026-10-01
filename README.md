# 饲料配方优化后端服务

把原料库、配方规格与每一次 LP 优化结果统一管理的后端服务。求解器为从零
实现的两阶段修正单纯形（仅依赖 NumPy），提供可逐条代入检验的最优性证明、
影子价格、检验数、价格灵敏度区间与不可行冲突解释。

## 快速开始

```bash
docker build -t feedopt .
docker run -p 8000:8000 -v $(pwd)/data:/data feedopt
```

本地开发：

```bash
pip install -r requirements-dev.txt
pytest
uvicorn app.api.app:app --reload
```

## 典型工作流

```bash
# 1. 维护原料草稿
curl -X PUT localhost:8000/ingredients/corn -H 'content-type: application/json' -d '{
  "ingredient_id":"corn","name":"玉米","price":0.30,"dry_matter":0.88,
  "nutrients":{"CP":8.0,"ME":3.35,"Ca":0.02,"P":0.27}}'
curl -X PUT localhost:8000/ingredients/sbm -H 'content-type: application/json' -d '{
  "ingredient_id":"sbm","name":"豆粕","price":0.50,"dry_matter":0.89,
  "nutrients":{"CP":44.0,"ME":2.6,"Ca":0.30,"P":0.65}}'

# 2. 发布原料库版本
curl -X POST localhost:8000/library/publish -d '{"note":"第40周报价"}'

# 3. 新建配方（修改走 PUT，自动产生新版本）
curl -X POST localhost:8000/formulas -H 'content-type: application/json' -d '{
  "name":"肉鸡1号",
  "spec":{
    "ingredients":[{"ingredient_id":"corn","min":0,"max":1},
                   {"ingredient_id":"sbm","min":0,"max":1}],
    "nutrient_bounds":{"CP":{"min":18.0}},
    "ratios":[]}}'

# 4. 即时优化（返回用量/成本/影子价格/检验数/自证/灵敏度）
curl -X POST localhost:8000/formulas/<id>/optimize -d '{}'

# 5. 周一换价：改草稿 -> 发布新版本 -> 提交批量作业（立即返回作业号）
curl -X POST localhost:8000/jobs/reprice -d '{"library_version":2}'
curl localhost:8000/jobs/<job_id>
curl -X POST localhost:8000/jobs/<job_id>/cancel

# 6. 历史与两个结果版本对比
curl localhost:8000/formulas/<id>/results
curl localhost:8000/formulas/<id>/compare/<optA>/<optB>
```

## 优化结果结构

`POST /formulas/{id}/optimize` 返回：

- `cost` / `dual_objective` / `relative_gap`（要求 < 1e-9）；
- `ingredients[]`：每种原料用量、检验数、状态（basic/at_lower/at_upper）；
- `constraints[]`：每条营养/比值/上下界/总量约束的左端值、松驰量、
  `active` 是否起作用、`shadow_price` 影子价格；
- `certificate`：`primal_feasible / dual_feasible /
  complementary_slackness` 三个布尔值与各项最大违例，以及逐列
  `x_j·r_j` 明细，可独立代入复核；
- `sensitivity[]`：每种原料价格区间 `[price_low, price_high]`，区间内
  最优配方组成不变；
- `basis` / `structure_key`：热启动所需的最优基与问题指纹。

不可行时 HTTP 409，`conflicts[]` 给出至少一组可解释冲突（如
“营养 CP 下限 60 高于各原料在添加上限内能达到的最大值 44”），
同时该次结果以 `infeasible` 状态入历史。

## 测试

```bash
pytest -q
```

覆盖：两原料手算算例；最优性三条件与对偶间隙；全体单价同乘正数时用量不变、
成本与影子价格同比放大；放松起作用约束成本不升（且一阶变化量等于影子价格）；
价格在灵敏度区间内组成不变、出区间后改变；不可行冲突说明（营养、上下限总量）；
热启动与冷启动成本一致（相对差 < 1e-9）及基失效回退；作业取消不留半批、
运行期版本锁定、同配方并发不互相覆盖；全部带字段名的校验错误。

## 设计取舍

详见 [`docs/DESIGN.md`](docs/DESIGN.md)，核心两条：

1. **防循环/确定性**：Bland 规则进基出基，同一输入永远同一最优基。
2. **热启动**：仅在问题结构（含右端项）相同且旧基原始可行时复用并跳过
   第一阶段；营养矩阵或约束变化导致基失效时自动回退冷启动——热启动只影响
   速度，不影响答案。
