import numpy as np
# Tesla powerwall has a power of 5.8kW without sunlight
# therefore, a maximum energy of 5.8 * 1/4 = 1,45kWh can be charged or discharged from the battery
# in a 15 minute interval
#MAX_BATTERY_ENERGY = 5.8 * 1 / 4


class Env:
    def __init__(self, df, full_battery_capacity=20, max_energy=5.8*1/4, n_days=2, n_steps=1000, low=0, high=30000, test=False):
        self.amount_paid = None
        self.market_price = None
        self.energy_consumption = None
        self.energy_generation = None
        self.ev_consumption = None
        self.current_battery_capacity = None
        self.time_of_day = None
        self.pos = None
        self.df = df
        self.n_steps = n_steps
        self.window_size = 24 * 4 * n_days
        self.low = low
        self.high = high
        self.test = test

        self.full_battery_capacity = full_battery_capacity
        self.max_energy = max_energy
        self.current_step = 0
        self.history = []
        self.reset(0)
        self.state = self.next_observation()

        # query for max and min values in dataframe (used for normalization)
        self.maxs = self.df.max()
        self.mins = self.df.min()

    def reset(self, seed):
        self.current_step = 0
        np.random.seed(seed)
        if self.test:
            self.pos = self.low
            #self.pos = self.pos - (self.pos % self.n_steps)
        else:
            self.pos = np.random.randint(self.low + self.window_size + self.n_steps, self.high - self.n_steps)
            self.pos = self.pos - (self.pos % self.n_steps)  # = 1 week
        self.current_battery_capacity = np.array([0] * self.window_size)
        self.energy_generation = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Generation'])
        self.energy_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Consumption'])
        self.ev_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'EV_Consumption'])
        self.market_price = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1, 'SMP'])
        self.amount_paid = np.array([0] * self.window_size)
        self.time_of_day = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Time_of_Day'])
        self.history = []
        return self.next_observation()

    def next_observation(self):
        return np.array([
            [self.current_step] * self.window_size,
            self.current_battery_capacity,
            self.energy_generation,
            self.energy_consumption,
            self.ev_consumption,
            self.market_price,
            self.amount_paid,
            self.time_of_day
        ])

    def next_observation_normalized(self):
        current_battery_capacity = self.current_battery_capacity / self.full_battery_capacity
        energy_generation = (self.energy_generation - self.mins['Energy_Generation']) / (
                self.maxs['Energy_Generation'] - self.mins['Energy_Generation'])
        energy_consumption = (self.energy_consumption - self.mins['Energy_Consumption']) / (
                self.maxs['Energy_Consumption'] - self.mins['Energy_Consumption'])
        ev_consumption = (self.ev_consumption - self.mins['EV_Consumption']) / (
                self.maxs['EV_Consumption'] - self.mins['EV_Consumption'])
        
        if self.maxs['SMP'] - self.mins['SMP'] == 0:
            market_price = self.market_price
        else:
            market_price = ((self.market_price - self.mins['SMP']) / (self.maxs['SMP'] - self.mins['SMP']))
        # amount paid = [market price] * [MWh bought from the grid]. Therefore max possible value equals
        # [max market price] * [max consumption]
        # and min possible value is 0 (if we take all the energy from the battery
        amount_paid = (self.amount_paid / (self.maxs['SMP'] * self.maxs['Energy_Consumption']))
        time_of_day = self.time_of_day
        return np.array([
            current_battery_capacity,
            energy_generation,
            energy_consumption,
            ev_consumption,
            market_price,
            amount_paid,
            time_of_day
        ]).flatten()

    def step(self, action):
        self.take_action(action)

        self.current_step += 1

        self.energy_generation = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Generation'])
        self.energy_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Consumption'])
        self.ev_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'EV_Consumption'])
        self.market_price = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1, 
            'SMP'])
        self.time_of_day = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Time_of_Day'])

        reward = -self.amount_paid[self.window_size - 1]
        terminated = self.current_step >= self.n_steps
        obs = self.next_observation()
        self.history.append(obs[:, self.window_size - 1])
        return obs, reward, terminated

    def take_action(self, action):
        # action = 0 -> charge battery from the grid, power house from the grid
        # action = 1 -> sell battery charge to the grid, power house from the battery
        # action = 2 -> just power house from the grid
        # action = 3 -> just power house from battery (if possible, else power from grid)
        # action = 4 -> charge battery from the grid, power house from the grid, charge ev from the grid
        # action = 5 -> sell battery charge to the grid, power house from the battery, charge ev from the battery
        consumption = self.energy_consumption[self.window_size - 1]
        ev_consumption = self.ev_consumption[self.window_size - 1]
        generation = self.energy_generation[self.window_size - 1]
        smp = self.market_price[self.window_size - 1]
        capacity = self.current_battery_capacity[self.window_size - 1]
        new_capacity = None
        amount_paid_now = None
        if action == 0:
            house_powering_price = consumption * smp
            battery_charged_from_grid = min(self.full_battery_capacity - capacity, self.max_energy)
            battery_charging_price = battery_charged_from_grid * smp
            new_capacity = min(capacity + battery_charged_from_grid + generation, self.full_battery_capacity)
            amount_paid_now = house_powering_price + battery_charging_price
        elif action == 1:
            house_powered_from_battery = min(capacity, self.max_energy, consumption)
            house_powered_from_grid = consumption - house_powered_from_battery
            house_powering_price = house_powered_from_grid * smp
            new_capacity = capacity - house_powered_from_battery
            battery_energy_to_sell = min(new_capacity, self.max_energy - house_powered_from_battery)
            new_capacity = min(new_capacity - battery_energy_to_sell + generation, self.full_battery_capacity)
            battery_discharging_profit = battery_energy_to_sell * smp
            amount_paid_now = house_powering_price - battery_discharging_profit
        elif action == 2:
            amount_paid_now = consumption * smp
            new_capacity = min(capacity + generation, self.full_battery_capacity)
        elif action == 3:
            house_powered_from_battery = min(capacity, self.max_energy, consumption)
            house_powered_from_grid = consumption - house_powered_from_battery
            amount_paid_now = house_powered_from_grid * smp
            new_capacity = min(capacity - house_powered_from_battery + generation, self.full_battery_capacity)
        elif action == 4:
            # charge car from grid as well
            house_powering_price = (consumption + ev_consumption) * smp
            battery_charged_from_grid = min(self.full_battery_capacity - capacity, self.max_energy)
            battery_charging_price = battery_charged_from_grid * smp
            new_capacity = min(capacity + battery_charged_from_grid + generation, self.full_battery_capacity)
            amount_paid_now = house_powering_price + battery_charging_price
        elif action == 5:
            # charge car from battery as well
            house_powered_from_battery = min(capacity, self.max_energy, consumption + ev_consumption)
            house_powered_from_grid = consumption + ev_consumption - house_powered_from_battery
            house_powering_price = house_powered_from_grid * smp
            new_capacity = capacity - house_powered_from_battery
            battery_energy_to_sell = min(new_capacity, self.max_energy - house_powered_from_battery)
            new_capacity = min(new_capacity - battery_energy_to_sell + generation, self.full_battery_capacity)
            battery_discharging_profit = battery_energy_to_sell * smp
            amount_paid_now = house_powering_price - battery_discharging_profit

        self.amount_paid = np.concatenate([self.amount_paid[1:self.window_size], [amount_paid_now]])
        self.current_battery_capacity = np.concatenate(
            [self.current_battery_capacity[1:self.window_size], [new_capacity]])

    def render(self):
        return None
        # print('Step:', self.current_step, '| Current battery capacity:', self.current_battery_capacity, '| Amount paid:', self.amount_paid)
