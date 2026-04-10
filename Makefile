PYTHON  ?= python3
TICKERS ?= data/tickers/sp100.txt
STRATEGY ?= sharpe
LOOKBACK ?= 60
TOP ?= 20
START ?= 2019-01-01
END ?= 2024-12-31
REBALANCE ?= 21

.PHONY: help install run backtest all-strategies clean reset

help:
	@echo "available targets:"
	@echo "  make install         install python dependencies"
	@echo "  make run             run the screener once with STRATEGY=$(STRATEGY)"
	@echo "  make backtest        run screener with rolling backtest"
	@echo "  make all-strategies  run backtest for every strategy"
	@echo "  make clean           remove outputs and plot artefacts"
	@echo "  make reset           clean plus clear the price cache"
	@echo ""
	@echo "variables (override like 'make run STRATEGY=momentum'):"
	@echo "  TICKERS   = $(TICKERS)"
	@echo "  STRATEGY  = $(STRATEGY)"
	@echo "  LOOKBACK  = $(LOOKBACK)"
	@echo "  TOP       = $(TOP)"
	@echo "  START     = $(START)"
	@echo "  END       = $(END)"
	@echo "  REBALANCE = $(REBALANCE)"

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m src.screener \
		--tickers $(TICKERS) \
		--strategy $(STRATEGY) \
		--lookback $(LOOKBACK) \
		--top $(TOP) \
		--start $(START) \
		--end $(END) \
		--verbose

backtest:
	$(PYTHON) -m src.screener \
		--tickers $(TICKERS) \
		--strategy $(STRATEGY) \
		--lookback $(LOOKBACK) \
		--top $(TOP) \
		--start $(START) \
		--end $(END) \
		--backtest \
		--rebalance $(REBALANCE) \
		--verbose

all-strategies:
	$(PYTHON) -m src.screener \
		--tickers $(TICKERS) \
		--strategy all \
		--lookback $(LOOKBACK) \
		--top $(TOP) \
		--start $(START) \
		--end $(END) \
		--backtest \
		--rebalance $(REBALANCE) \
		--verbose

clean:
	rm -f output/*.csv output/*.json
	rm -f execution_artefacts/*.png execution_artefacts/*.gif execution_artefacts/*.log

reset: clean
	rm -rf data/cache
