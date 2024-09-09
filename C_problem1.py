import numpy as np
import random
import pandas as pd

class GeneticAlgorithm:
    def __init__(self, num_individuals, num_years, num_plots, num_crops_per_plot):
        self.num_individuals = num_individuals  # 种群中个体数量
        self.num_years = num_years  # 总的年份数
        self.num_plots = num_plots  # 地块总数
        self.num_crops_per_plot = num_crops_per_plot  # 每个地块的作物种类数量
        self.best_fitness = float('-inf')
        self.best_individual = None
        self.max_areas = np.array([
            80, 55, 35, 72, 68, 55, 60, 46, 40, 28, 25, 86, 55, 44, 50, 25, 60, 45, 
            35, 20, 15, 13, 15, 18, 27, 20, 15, 10, 14, 6, 10, 12, 22, 20, 0.6, 0.6, 
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 
            0.6, 0.6, 0.6, 0.6
        ])

        # 初始化亩产量、种植成本、销售单价矩阵
        self.yield_matrix = np.array([
            [400, 500, 400, 350, 415, 800, 1000, 400, 630, 525, 110, 3000, 2200, 420, 525] + [0] * (41 - 15),
            [380, 475, 380, 330, 395, 760, 950, 380, 600, 500, 105, 2850, 2100, 400, 500] + [0] * (41 - 15),
            [360, 450, 360, 315, 375, 720, 900, 360, 570, 475, 100, 2700, 2000, 380, 475] + [0] * (41 - 15),
            [0] * 15 + [500, 3000, 2000, 3000, 2000, 2400, 6400, 2700, 2400, 3300, 3700, 4100, 3200, 12000, 4100, 1600, 10000, 5000, 5500] + [0] * (41 - 34),
            [0] * 16 + [3600, 2400, 3600, 2400, 3000, 8000, 3300, 3000, 4000, 4500, 5000, 4000, 15000, 5000, 2000, 12000, 6000, 6600, 5000, 4000, 3000, 5000, 4000, 10000, 1000],
            [0] * 16 + [3200, 2200, 3200, 2200, 2700, 7200, 3000, 2700, 3600, 4100, 4500, 3600, 13500, 4500, 1800, 11000, 5400, 6000] + [0] * (41 - 34)
        ])

        self.cost_matrix = np.array([
            [400, 400, 350, 350, 350, 450, 500, 360, 400, 360, 350, 1000, 2000, 400, 350] + [0] * (41 - 15),
            [400, 400, 350, 350, 350, 450, 500, 360, 400, 360, 350, 1000, 2000, 400, 350] + [0] * (41 - 15),
            [400, 400, 350, 350, 350, 450, 500, 360, 400, 360, 350, 1000, 2000, 400, 350] + [0] * (41 - 15),
            [0] * 15 + [680, 2000, 1000, 2000, 2000, 2000, 2000, 2300, 1600, 2400, 2900, 1600, 1600, 2900, 1600, 1000, 4100, 2000, 900] + [0] * (41 - 34),
            [0] * 16 + [2400, 1200, 2400, 2400, 2400, 2400, 2700, 2000, 3000, 3500, 2000, 2000, 3500, 2000, 2000, 5500, 2750, 1100, 2000, 500, 500, 3000, 2000, 10000, 10000],
            [0] * 16 + [2640, 1320, 2640, 2640, 2640, 2640, 3000, 2200, 3300, 3850, 2200, 2200, 3850, 2200, 1300, 5500, 2750, 1100] + [0] * (41 - 34)
        ])

        self.price_matrix = np.array([
            [4.00, 8.5, 9.00, 8.00, 7.5, 4, 3.5, 7.5, 6.5, 8.5, 50.00, 2.0, 4.00, 6.00, 4.0] + [0] * (41 - 15),
            [4.0, 8.50, 9.00, 8.00, 7.5, 4.00, 3.5, 7.50, 6.5, 8.5, 50.00, 2.00, 4.00, 6.00, 4.0] + [0] * (41 - 15),
            [4.00, 8.50, 9.00,8.00, 7.50, 4.00, 3.5, 7.50, 6.5, 8.5, 50.00, 2.00, 4.00, 6.00, 4.0] + [0] * (41 - 15),
            [0] * 15 + [8.00, 9.00, 8.00, 8.00, 4.50, 7.50, 6.00, 6.70, 6.50, 6.00, 7.50, 6.00, 6.50, 8.00, 6.00, 8.50, 6.00, 5.00,4.80] + [0] * (41 - 34),
            [0] * 16 + [9.00, 8.00, 8.00, 4.50, 7.50, 6.00, 6.70, 6.50, 6.00, 7.50, 6.00, 6.50, 8.00, 6.00, 8.50, 6.00, 5.00,4.80, 3.00, 3.00, 4.00, 65.00, 20.00, 18.00, 120.00],
            [0] * 16 + [10.80, 9.60, 9.60, 5.40, 9.00, 7.20, 8.00, 7.80, 7.20, 9.00, 7.20, 7.80, 9.60, 7.20, 10.20, 7.20, 6.00, 5.80] + [0] * (41 - 34)
        ])
        self.legume_indices = list(range(1, 6)) + list(range(17, 21))  # 豆类作物的索引
        #每种作物的销量
        self.xiaoliang = np.array([
            147, 46, 60, 96, 25, 222, 135, 
            185, 50, 25, 15, 13, 18, 35, 
            20, 42, 11.8, 13.2, 1.8, 15, 14.9, 
            6.9, 0.3, 0.9, 0.9, 0.9, 0.9, 10.9, 
            0.9, 0.6, 0.6, 0.3, 0.3, 0.3, 
            30, 25, 12, 1.8, 1.8, 1.8, 4.2
        ])

        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.num_individuals):
            individual = np.zeros((self.num_years, self.num_plots, self.num_crops_per_plot, 2))
            for t in range(self.num_years):
                for i in range(self.num_plots):
                    if self.is_single_season_plot(i):
                        crop_areas = np.zeros(self.num_crops_per_plot)
                        crop_areas = self.apply_crop_restrictions(i, crop_areas, 0,individual,t)  # 单季作物只限制第一季
                        individual[t, i, :, 0] = crop_areas
                        individual[t, i, :, 1] = 0
                    else:
                        crop_areas_first_season = np.zeros(self.num_crops_per_plot)
                        crop_areas_second_season = np.zeros(self.num_crops_per_plot)
                        crop_areas_first_season = self.apply_crop_restrictions(i, crop_areas_first_season, 0,individual,t)  # 第一季
                        crop_areas_second_season = self.apply_crop_restrictions(i, crop_areas_second_season, 1,individual,t)  # 第二季
                        individual[t, i, :, 0] = crop_areas_first_season
                        individual[t, i, :, 1] = crop_areas_second_season
    
            
            # 检查并满足作物不连续种植要求
            individual = self.ensure_no_consecutive_cropping(individual)
            # 检查并满足豆类种植要求
            individual = self.ensure_legumes(individual)  
            # 在检查豆类种植要求后，再次限制面积
            for t in range(self.num_years):
                for i in range(self.num_plots):
                    individual[t, i, :, 0] = self.limit_area(individual[t, i, :, 0], i)  # 限制第一季面积
                    individual[t, i, :, 1] = self.limit_area(individual[t, i, :, 1], i)  # 限制第二季面积  
            population.append(individual)
        return population


    def limit_area(self, crop_areas, plot_index):
        total_area = np.sum(crop_areas)
        max_area = self.max_areas[plot_index]
        if total_area > max_area:
            scale_factor = max_area / total_area
            crop_areas *= scale_factor
        return crop_areas

    def is_single_season_plot(self, plot_index):
        # 根据地块索引判断地块是否为单季作物地块
        return plot_index < 27
    def get_plot_category(self,plot_index):
        if plot_index < 6:
            return 0
        elif plot_index < 6 + 14:
            return 1
        elif plot_index < 6 + 14 + 6:
            return 2
        elif plot_index < 6 + 14 + 6 + 8:
            return 3
        elif plot_index < 6 + 14 + 6 + 8 + 16:
            return 4
        elif plot_index < 6 + 14 + 6 + 8 + 16 + 4:
            return 5
        else:
            raise ValueError("地块索引超出范围")
    def apply_crop_restrictions(self, plot_index, crop_areas, season,individual,t):
        max_area = self.max_areas[plot_index]  # 获取当前地块的最大面积
        if plot_index < 6:  # 第1类，干旱地
            crop_areas[:15] = 0  # 确保只选择一种作物
            selected_crop = random.choice(range(15))  # 随机选择1-15中的一种
            crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积
        elif plot_index < 20:  # 第2类，梯田
            crop_areas[:15] = 0  # 确保只选择一种作物
            selected_crop = random.choice(range(15))  # 随机选择1-15中的一种
            crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积
        elif plot_index < 26:  # 第3类，坡地
            crop_areas[:15] = 0  # 确保只选择一种作物
            selected_crop = random.choice(range(15))  # 随机选择1-15中的一种
            crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积
        elif plot_index < 34:  # 第4类，水浇地
            if season == 0:  # 第一季
                # 随机选择16-34中的一种作物
                selected_crop = random.choice(range(16, 35))
                
                # 判断第一季是否种植稻米（16号作物）
                if selected_crop == 16:
                    # 如果是稻米，只分配面积给稻米，第二季不种其他作物
                    crop_areas[1:16] = 0  # 只能选择16（稻米）
                    crop_areas[16] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积给稻米
                    crop_areas[17:] = 0  # 确保第二季不种作物
                else:
                    # 如果不是稻米（选择了17-34中的作物），第一季种该作物
                    crop_areas[:16] = 0  # 确保16号及以下作物不种
                    crop_areas[35:] = 0  # 确保34号及以上作物不种
                    crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积给选定作物
            
            else:  # 第二季，只有在第一季未种稻米时，种植17-34中的一种作物
                if individual[t, plot_index, 16, 0] == 0:  # 检查第一季是否种了稻米
                    crop_areas[:17] = 0  # 确保只能种17-34
                    crop_areas[35:] = 0
                    selected_crop = random.choice(range(17, 35))  # 随机选择17-34中的一种
                    crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积
                else:
                    # 如果第一季种了稻米，第二季不种作物
                    crop_areas[:] = 0
        elif plot_index < 50:  # 第5类，普通大棚
            if season == 0:  # 第一季，种植17-34中的一种作物
                crop_areas[:17] = 0  # 确保只能种17-34
                selected_crop = random.choice(range(17, 35))  # 随机选择17-34中的一种
                crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积
            else:  # 第二季，种植38-41中的一种作物
                crop_areas[:38] = 0  # 确保只能种38-41
                selected_crop = random.choice(range(38, 41))  # 随机选择38-41中的一种
                crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积
        elif plot_index < 54:  # 第6类，智慧大棚
            crop_areas[:17] = 0  # 确保只能种17-34
            selected_crop = random.choice(range(17, 35))  # 随机选择17-34中的一种
            crop_areas[selected_crop] = np.random.uniform(max_area * 0.8, max_area * 1.2)  # 分配面积
        return crop_areas
    def ensure_legumes(self, individual):
    # 从第三年开始循环
        for t in range(2, self.num_years):
            for i in range(self.num_plots):
                # 检查当前年及前两年是否种植了豆类作物
                legume_found = False
                for offset in [-2, -1, 0]:
                    year = t + offset
                    if 0 <= year < self.num_years:
                        if any(crop in self.legume_indices for crop in np.nonzero(individual[year, i, :, 0])[0]) or \
                           any(crop in self.legume_indices for crop in np.nonzero(individual[year, i, :, 1])[0]):
                            legume_found = True
                            break
                
                # 如果三年内没有种植豆类作物，则在当前年随机选择一个季节种植豆类
                if not legume_found:
                    # 确定块地类型并选择合适的豆类
                    if i < 26:  # 类别1-3
                        available_legumes = [c for c in self.legume_indices if 1 <= c <= 5]
                        # 只能在第一季种植作物
                        season = 0
                    elif 26 <= i < 34:  # 类别4
                        if np.any(individual[t, i, 16, 0] > 0):
                            # 如果第一季种了16号作物，则第二季不能种植任何作物
                            individual[t, i, :, 1] = 0
                            continue
                        available_legumes = [c for c in self.legume_indices if 17 <= c <= 19]
                        season = random.choice([0, 1])
                    elif 34 <= i < 50:  # 类别5
                        available_legumes = [c for c in self.legume_indices if  17 <= c <= 19]
                        # 只能在第一季种植作物
                        season = 0
                    else:  # 类别6
                        available_legumes = [c for c in self.legume_indices if  17 <= c <= 19]
                        season = random.choice([0, 1])
                    
                    # 随机选择一个豆类并设置
                    selected_legume = random.choice(available_legumes)
                    
                    # 清空该季的原有作物
                    individual[t, i, :, season] = 0
                    # 设置豆类作物
                    individual[t, i, selected_legume, season] = np.random.uniform(self.max_areas[i] * 0.8, self.max_areas[i] * 1.2)
        
        return individual

    def ensure_no_consecutive_cropping(self, individual):
        for t in range(self.num_years):
            for i in range(self.num_plots):
                for year in range(1, self.num_years):
                    for season in range(2):
                        prev_crop_areas = individual[year - 1, i, :, season]
                        curr_crop_areas = individual[year, i, :, season]
                        prev_crop = np.argmax(prev_crop_areas)
                        curr_crop = np.argmax(curr_crop_areas)
                        if prev_crop == curr_crop:
                            new_crop = random.choice(list(set(range(self.num_crops_per_plot)) - {curr_crop}))
                            if season == 0:
                                individual[year, i, new_crop, season] = np.random.uniform(self.max_areas[i] * 0.8, self.max_areas[i] * 1.2)
                            else:
                                individual[year, i, new_crop, season] = np.random.uniform(self.max_areas[i] * 0.8, self.max_areas[i] * 1.2)
                            individual[year, i, curr_crop, season] = 0
        return individual
    def calculate_fitness(self, individual):
        total_profit = 0
       
        # 遍历每一年
        for t in range(self.num_years):
            yearly_profit = 0
            # 初始化用于跟踪每种作物累计产量的数组
            accumulated_yield = np.zeros(self.num_crops_per_plot)
            # 遍历每个地块
            for i in range(self.num_plots):
                # 遍历每个季度
                for s in range(2):
                    # 获取当前地块和季度的作物面积
                    crop_areas = individual[t, i, :, s]
                    category = self.get_plot_category(i)
                    # 获取当前地块和季度的亩产量、种植成本和销售单价
                    yields = self.yield_matrix[category]
                    costs = self.cost_matrix[category]
                    prices = self.price_matrix[category]
                    # 计算每种作物的利润
                    for j in range(self.num_crops_per_plot):
                        # 当前作物在该地块和季度的种植面积
                        area = crop_areas[j]               
                        if area == 0:
                            continue                
                        # 计算该作物的总面积
                        total_yield = area
                        if accumulated_yield[j] >= self.xiaoliang[j]*1.5:
                        # 如果累计产量超过预期销量，只有成本，没有销售收入
                            cost = area * costs[j]
                            yearly_profit -= cost  # 直接减去成本，利润为负
                        else:
                            sellable_yield = min(total_yield, self.xiaoliang[j]*1.5 - accumulated_yield[j])
                                # 计算销售收入
                            revenue = sellable_yield * prices[j]* yields[j]
                                # 更新累计产量
                            accumulated_yield[j] += sellable_yield
                                # 计算种植成本
                            cost = area * costs[j]
                                # 计算作物的利润
                            profit = revenue - cost
        
                                # 将作物利润加到每年的总利润中
                            yearly_profit += profit
    
            # 将每年的总利润加到总利润中
            total_profit += yearly_profit
        return total_profit

    
        return total_profit
    def print_population(self):
            for i, individual in enumerate(self.population):
                print(f"个体 {i + 1}:")
                for t in range(self.num_years):
                    print(f"  第 {t + 1} 年:")
                    for i_plot in range(self.num_plots):
                        print(f"    地块 {i_plot + 1}:")
                        if self.is_single_season_plot(i_plot):
                            print(f"      作物面积 (第一季): {individual[t, i_plot, :, 0]}")
                        else:
                            print(f"      作物面积 (第一季): {individual[t, i_plot, :, 0]}")
                            print(f"      作物面积 (第二季): {individual[t, i_plot, :, 1]}")
                print()
    def print_top_individual_fitnesses(self, num_top_individuals=100):
        # 计算所有个体的适应度
        fitnesses = [self.calculate_fitness(individual) for individual in self.population]

        # 获取前 num_top_individuals 个个体的适应度
        top_indices = np.argsort(fitnesses)[-num_top_individuals:]
        top_fitnesses = [fitnesses[i] for i in top_indices]

        print("前 {} 个个体的适应度:".format(num_top_individuals))
        for i, fitness in enumerate(top_fitnesses):
            print("个体 {}: 适应度 = {}".format(i + 1, fitness))
    def select_parents(self):
        # 计算所有个体的适应度值
        fitness_values = [(ind, self.calculate_fitness(ind)) for ind in self.population]
        
        # 按照适应度值从高到低排序
        fitness_values.sort(key=lambda x: x[1], reverse=True)
        
        # 选取适应度前10%的个体作为候选父代
        top_10_percent_count = max(1, int(len(fitness_values) * 0.05))
        top_individuals = [ind for ind, _ in fitness_values[:top_10_percent_count]]
        
        # 随机选择两个父代
        parent1 = top_individuals[np.random.randint(len(top_individuals))]
        parent2 = top_individuals[np.random.randint(len(top_individuals))]
        
        return parent1, parent2

    def crossover(self, parent1, parent2):
        # 单点交叉在每个地块类别内进行
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)
        
        # 分类交叉处理
        for plot_index in range(self.num_plots):
            category = self.get_plot_category(plot_index)
            if category in [0, 1, 2]:  # 前三类地块
                year_point = np.random.randint(0,self.num_years)
                crossover_point = np.random.randint(0, 16)
                # 对每个地块在分类范围内进行交叉
                offspring1[year_point,plot_index,crossover_point:,0] = np.copy(parent2[year_point,plot_index,crossover_point:,0])
                offspring2[year_point,plot_index,crossover_point:,0] = np.copy(parent1[year_point,plot_index,crossover_point:,0])
                # 对 offspring1 和 offspring2 进行检查，确保没有两个种植面积同时大于0
                if np.count_nonzero(offspring1[year_point, plot_index, :, 0] > 0) > 1:
                    # 随机选择大于0的作物，并将其置为0
                    non_zero_indices = np.where(offspring1[year_point, plot_index, :, 0] > 0)[0]
                    zero_index = np.random.choice(non_zero_indices)
                    offspring1[year_point, plot_index, zero_index, 0] = 0
                
                if np.count_nonzero(offspring2[year_point, plot_index, :, 0] > 0) > 1:
                    # 随机选择大于0的作物，并将其置为0
                    non_zero_indices = np.where(offspring2[year_point, plot_index, :, 0] > 0)[0]
                    zero_index = np.random.choice(non_zero_indices)
                    offspring2[year_point, plot_index, zero_index, 0] = 0

            elif category in [3, 4, 5]:  # 后三类地块
                year_point = np.random.randint(0,self.num_years)
                crossover_point = np.random.randint(16, 41)
                # 对每个地块在分类范围内进行交叉
                offspring1[year_point,plot_index,crossover_point:,0] = np.copy(parent2[year_point,plot_index,crossover_point:,0])
                offspring2[year_point,plot_index,crossover_point:,0] = np.copy(parent1[year_point,plot_index,crossover_point:,0])
    
        # 交叉后进行豆类和连作检验
        offspring1 = self.ensure_no_consecutive_cropping(offspring1)
        offspring1 = self.ensure_legumes(offspring1)
    
        offspring2 = self.ensure_no_consecutive_cropping(offspring2)
        offspring2 = self.ensure_legumes(offspring2)
        #限制面积
        for t in range(self.num_years):
            for i in range(self.num_plots):
                offspring1[t, i, :, 0] = self.limit_area(offspring1[t, i, :, 0], i)  # 限制第一季面积
                offspring2[t, i, :, 1] = self.limit_area(offspring2[t, i, :, 1], i)  # 限制第二季面积  
        
        return offspring1, offspring2

    def mutation(self, individual, mutation_rate=0.001):
        # 遍历个体并随机进行变异
        for t in range(self.num_years):
            for i in range(self.num_plots):
                for s in range(2):
                    for j in range(self.num_crops_per_plot):
                        if np.random.rand() < mutation_rate:
                            # 如果当前作物的种植面积非零，则进行变异
                            if individual[t, i, j, s] > 0:
                                # 将面积乘以一个随机数 (0-1 之间)
                                scale_factor = np.random.rand()*2
                                individual[t, i, j, s] *= scale_factor
                                self.limit_area(individual[t, i, j, s], i)
        return individual


    def run_algorithm(self, num_generations):
        for generation in range(num_generations):
            new_population = []
            
            # 创建新一代种群
            for _ in range(self.num_individuals // 2):
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)
                new_population.append(offspring1)
                new_population.append(offspring2)
            
            # 更新种群
            self.population = new_population
            
            # 记录当前代的最优个体
            for individual in self.population:
                fitness_value = self.calculate_fitness(individual)
                if fitness_value > self.best_fitness:
                    self.best_fitness = fitness_value
                    self.best_individual = individual

            print(f"Generation {generation + 1}: Best Fitness = {self.best_fitness}")
        
        # 输出全局最优个体及其适应度
        print("Final Best Individual and Fitness:")
        print("Best Individual:", self.best_individual)
        print("Best Fitness:", self.best_fitness)
        self.save_best_individual_to_excel()


    def save_best_individual_to_excel(self):
        # 用于存储数据的列表
        rows = []
    
        # 遍历最优个体，将信息按指定格式保存到列表中
        for t in range(self.num_years):
            # 插入年份标题
            rows.append([f"第 {t + 1} 年", "", ""])  # 确保这一行只有三列
            for i_plot in range(self.num_plots):
                if self.is_single_season_plot(i_plot):
                    # 只记录第一季作物的面积，将面积转换为字符串以确保列一致
                    crop_areas_first_season = self.best_individual[t, i_plot, :, 0]
                    crop_areas_str = ', '.join(map(str, crop_areas_first_season))
                    rows.append([f"  地块 {i_plot + 1}", "第一季", crop_areas_str])
                else:
                    # 记录两季作物的面积，并确保作物面积格式一致
                    crop_areas_first_season = self.best_individual[t, i_plot, :, 0]
                    crop_areas_second_season = self.best_individual[t, i_plot, :, 1]
                    crop_areas_first_str = ', '.join(map(str, crop_areas_first_season))
                    crop_areas_second_str = ', '.join(map(str, crop_areas_second_season))
                    rows.append([f"  地块 {i_plot + 1}", "第一季", crop_areas_first_str])
                    rows.append(["", "第二季", crop_areas_second_str])
    
        # 将列表转换为 DataFrame
        df = pd.DataFrame(rows, columns=["年份/地块", "季节", "作物面积"])
        
        # 将 DataFrame 保存为 Excel 文件
        df.to_excel("best_individual_detailed.xlsx", index=False)
        
        print("Best individual saved in detailed format to best_individual_detailed.xlsx")

# 示例使用
ga = GeneticAlgorithm(num_individuals=50, num_years=7, num_plots=54, num_crops_per_plot=41)
ga.run_algorithm(num_generations=50)

